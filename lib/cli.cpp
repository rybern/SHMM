#include "libshmm.cpp"

/*
  non-continuous memory for emissions
  n_events - 1
  add row, col to emissions?
  handle permutation
 */

int main(int argc, char** argv) {
  int n_states = 2;
  int n_events = 5;
  int n_obs = 2;
  double emissions_[n_states][n_obs + 1] = {
    {0.9, 0.2, 0},
    {0.9, 0.2, 0},
    {0.1, 0.9, 0},
    {0.9, 0.2, 0},
    {0.9, 0.2, 0}
  };
  double* emissions[n_obs + 1];
  for (int i = 0; i < n_events; i++)
    emissions[i] = &emissions_[i][0];

  int n_triples = 8;
  DTriple triples[n_triples] = {
    {0, 1, 0.5},
    {0, 2, 0.5},
    {1, 1, 0.7},
    {1, 2, 0.3},
    {1, 3, 0.5},
    {2, 1, 0.3},
    {2, 2, 0.7},
    {2, 3, 0.5}
  };
  int permutation[n_states] = {0, 1};
  double posterior_arr[n_events * (n_states + 1)];
  shmm(n_triples, triples,
       n_states, n_obs, emissions,
       n_events, permutation,
       posterior_arr);

  for (int i=0; i< n_events * (n_states + 1); i++)
    cerr << "posterior array[" << i << "]: " << posterior_arr[i] << endl;

  //shmm_cli(argc, argv);
}

/*
int n_states = 3;
int n_events = 6;
int n_obs = 3;
double emissions[n_states][n_obs] = {
  {0.9, 0.2, 0.0},
  {0.9, 0.2, 0.0},
  {0.1, 0.9, 0.0},
  {0.9, 0.2, 0.0},
  {0.9, 0.2, 0.0},
  {0.0, 0.0, 1.0}
};
int n_triples = 8;
DTriple triples[n_triples] = {
  {0, 1, 0.5},
  {0, 2, 0.5},
  {1, 1, 0.7},
  {1, 2, 0.3},
  {1, 3, 0.5},
  {2, 1, 0.3},
  {2, 2, 0.7},
  {2, 3, 0.5}
};
int permutation[1];
double posterior_arr[(n_events - 1) * n_states];
*/
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

int shmm_cli(int argc, char *argv[])
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

  SparseVector<double> initial(n_states);
  for(std::vector<T>::iterator it = initial_list.begin(); it != initial_list.end(); ++it)
    initial.insert(it->col()) = it->value();
  initial /= initial.sum();

  //initial = initial.transpose();

  //MatrixXd emissions = load_csv<MatrixXd>(emissions_filepath);
  vector<VectorXd> emissions = load_emissions(emissions_filepath, permutation);
  int n_events = emissions.size();
  double posterior_arr[(n_events-1) * n_states];
  Map<MatrixXd> posterior(posterior_arr, n_events-1, n_states);
  //MatrixXd posterior(n_events-1, n_states);

  if (verbose) {
    // cerr << "trans: " << endl << trans.row(0) << endl;
    cerr << "trans(" << trans.rows() << ", " << trans.cols() << "):" << endl;
    for (int i = 0; i < trans.rows(); i++)
      cerr << " " << trans.row(i);
  }

  forward_backward ( initial, trans, emissions, &posterior, verbose );

  if (verbose)
    cerr << "posterior: " << endl << posterior << endl;

  const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
  ofstream posterior_file(posterior_filepath);
  posterior_file << posterior.format(CSVFormat);
}

