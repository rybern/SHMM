#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <armadillo>
#include <assert.h>

using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> T;

template<typename M>
M load_csv (const std::string & path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

std::vector<int> load_emission_permutation (const std::string & path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  assert(std::getline(indata, line));
  std::stringstream lineStream(line);
  std::string cell;
  std::vector<int> values;
  while (std::getline(lineStream, cell, ',')) {
    values.push_back(std::stoi(cell));
  }
  cerr << "returning permutation " << values.size() << endl;
  return values;
}

template<typename T>
std::vector<T> apply_permutation_dense(std::vector<int> perm, std::vector<T> vec) {
  if (perm.size() == 0) {
    return vec;
  } else {
    int len = perm.size();
    std::vector<T> res(len);
    for (int i = 0; i < len; i++)
      res[i] = vec[perm[i]];
    return res;
  }
}

std::vector<VectorXd> load_emissions (const std::string & path, std::vector<int> permutation) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<VectorXd> row_vec;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    std::vector<double> values;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    // to indicate that there is zero probability of this being the end token
    values = apply_permutation_dense(permutation, values);
    values.push_back(0);

    row_vec.push_back(Map<const VectorXd>(values.data(), values.size()));
  }
  uint cols = row_vec[0].size();
  VectorXd end_emissions = MatrixXd::Zero(cols,1).col(0);
  end_emissions.coeffRef(cols-1) = 1.0;
  row_vec.push_back(end_emissions);
  return row_vec;
}

/*
notes:
  reusing emission columns:
    permutation matrix
      doesn't store, but can't grow matrix
    map
      creates new view of existing data, not sure if repeats work
 */

int main(int argc, char *argv[])
{
  bool verbose = false;
  char* trans_filepath = argv[1];
  char* emissions_filepath = argv[2];
  char* posterior_filepath = argv[3];

  std::vector<int> permutation;
  if (argc > 4) {
    cerr << "Starting SHMM" << endl;
    char* permutation_filepath = argv[4];
    permutation = load_emission_permutation(permutation_filepath);
  } else if (argc < 4) {
    cerr << "Usage: " << argv[0] << " TRANS.st EMISSIONS.csv OUTPUT.csv [PERMUTATION.CSV]" << endl;
    exit(1);
  }

  std::vector<T> trans_list;
  std::vector<T> initial_list;

  std::ifstream infile(trans_filepath);
  std::string line;
  while (std::getline(infile, line))
  {
    // ignore # lines
    if (line.at(0) == '#')
      continue;

    std::istringstream iss(line);

    // int int float
    int r, c;
    double val;
    if (!(iss >> r >> c >> val)) { break; } // error

    if (r == 0) {
      initial_list.push_back(T(r,c-1,val));
    } else {
      trans_list.push_back(T(r-1,c-1,val));
    }
  }

  int max_row = 0;
  int max_col = 0;
  for(std::vector<T>::iterator it = initial_list.begin(); it != initial_list.end(); ++it) {
    int c = it->col();
    if (c > max_col)
      max_col = c;
  }
  for(std::vector<T>::iterator it = trans_list.begin(); it != trans_list.end(); ++it) {
    int r = it->row();
    int c = it->col();
    if (r > max_row)
      max_row = r;
    if (c > max_col)
      max_col = c;
  }
  cerr << "max_row" << max_row << "max_col" << max_col << endl;
  assert(max_row + 1 == max_col);
  int n_states = max_col + 1;

  trans_list.push_back(T(max_row+1, max_col, 1.0));

  SparseMatrix<double> trans(n_states, n_states);
  trans.setFromTriplets(trans_list.begin(), trans_list.end());

  cerr << "n states: " << n_states << endl;
  if (verbose) {
    cerr << "trans: " << endl;
    for (int i = 0; i < trans.rows(); i++)
      cerr << " " << trans.row(i);
  }

  SparseVector<double> initial(n_states);
  for(std::vector<T>::iterator it = initial_list.begin(); it != initial_list.end(); ++it)
    initial.insert(it->col()) = it->value();
  initial /= initial.sum();
  if (verbose) {
    cerr << "initial: " << endl << initial << endl;
  }

  //initial = initial.transpose();

  //MatrixXd emissions = load_csv<MatrixXd>(emissions_filepath);
  vector<VectorXd> emissions = load_emissions(emissions_filepath, permutation);
  int n_events = emissions.size();

  cerr << "n emissions[0]: " << emissions[0].size() << endl;
  cerr << "n events: " << n_events << endl;

  if (verbose) {
    cerr << "emissions: " << endl;
    for (int i = 0; i < emissions.size(); i++)
      cerr << " " << emissions[i].transpose() << endl;
  }


  assert(emissions[0].size() == n_states);

  std::vector<SparseVector<double>> forward(n_events+1);
  forward[0] = initial;
  for (int i = 1; i <= n_events; i ++) {
    forward[i] = (trans.transpose() * forward[i-1]).cwiseProduct(emissions[i-1]);
    forward[i] /= forward[i].sum();
    if(forward[i].sum() == 0)
      cerr << "forward sum zero at row" << i << endl;
  }

  if (verbose) {
    cerr << "forward: " << endl;
    for (int i = 0; i < forward.size(); i++)
      cerr << " " << forward[i];
  }

  std::vector<SparseVector<double>> backward(n_events+1);
  backward[n_events] = MatrixXd::Ones(n_states,1).col(0).sparseView();
  for (int i = n_events - 1; i >= 0; i --) {
    backward[i] = (trans * backward[i+1].cwiseProduct(emissions[i]));
    backward[i] /= backward[i].sum();
    if(backward[i].sum() == 0)
      cerr << "backward sum zero at row" << i << endl;
  }

  if (verbose) {
    cerr << "backward: " << endl;
    for (int i = 0; i < backward.size(); i++)
      cerr << " " << backward[i];
  }

  // use -1 to avoid reporting the end token
  MatrixXd posterior(n_events-1, n_states);
  for (int i = 0; i < n_events-1; i ++) {
    //SparseVector<double> row = forward[i].cwiseProduct(backward[i]);
    VectorXd row = forward[i+1].cwiseProduct(backward[i+1]);
    double s = row.sum();
    posterior.row(i) = row / s;
  }
  if (verbose)
    cerr << "posterior: " << endl << posterior << endl;

  const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
  ofstream posterior_file(posterior_filepath);
  posterior_file << posterior.format(CSVFormat);
}

/*
  std::vector<SparseVector<double>> posterior(n_events);
  for (int i = 0; i < n_events; i ++) {
  //SparseVector<double> row = forward[i].cwiseProduct(backward[i]);
  posterior[i] = forward[i].cwiseProduct(backward[i]);
  posterior[i] /= posterior[i].sum();
  }

  cerr << "posterior: " << endl;
  for (int i = 0; i < posterior.size(); i++)
  cerr << " " << posterior[i];

  const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
  ofstream posterior_file(posterior_filepath);
  for (int i = 0; i < posterior.size(); i++)
  posterior_file << posterior[i].format(CSVFormat);
  //posterior_file << posterior.format(CSVFormat);
  */
