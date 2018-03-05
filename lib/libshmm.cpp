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

/*
notes:
  reusing emission columns:
    permutation matrix
      doesn't store, but can't grow matrix
    map
      creates new view of existing data, not sure if repeats work
 */
void forward_backward ( SparseVector<double> initial,
                        SparseMatrix<double> trans,
                        vector<VectorXd> emissions,
                        Map<MatrixXd> *posterior,
                        int n_events, int *permutation_,
                        bool verbose);

struct DTriple {
  int row;
  int col;
  double val;
};

void addTriples(int n_triples, DTriple *triples, SparseMatrix<double> *trans, SparseVector<double> *initial);

int shmm(int n_triples, DTriple *triples,
         int n_states, int n_obs, double **emissions_aptr,
         int n_events, int *permutation,
         double *posterior_arr ) {
  cerr << "params: " << "n_triples=" << n_triples << ", n_states=" << n_states << ", n_obs=" << n_obs << ", n_events=" << n_events << endl;
  SparseMatrix<double> trans(n_states+1, n_states+1);
  SparseVector<double> initial(n_states+1);
  addTriples(n_triples, triples, &trans, &initial);

  cerr << "trans(" << trans.rows() << ", " << trans.cols() << "):" << endl;
  for (int i = 0; i < trans.rows(); i++)
    cerr << " " << trans.row(i);
  cerr << "init: " << initial;
  cerr << "permutation: " << permutation[0] << permutation[1] << permutation[2] << endl;

  const char* emissions_filepath = "test/emissions.csv";
  //std::vector<int> permutation;
  //permutation.assign(permutation_, permutation_ + n_states);
  //cerr << "perm: " << permutation;
  //vector<VectorXd> emissions = load_emissions(emissions_filepath, permutation);

  vector<VectorXd> emissions;
  for (int i = 0; i < n_events; i++) {
    //emissions.push_back(Map<const VectorXd>(emissions_aptr + i * (n_obs + 1), (n_obs + 1)));
    emissions.push_back(Map<const VectorXd>(emissions_aptr[i], (n_obs + 1)));
  }
  VectorXd end_emissions = MatrixXd::Zero(n_obs+1,1).col(0);
  end_emissions.coeffRef(n_obs) = 1.0;
  emissions.push_back(end_emissions);

  Map<MatrixXd> posterior(posterior_arr, n_events, n_states + 1);

  forward_backward ( initial, trans, emissions, &posterior, n_states+1, permutation, true );

  cerr << "posterior: " << endl << posterior << endl;
}

// this could definitely be done better.
// we iterate over a list to create an array, then iterate over an array to create a vector, then iterate over a vector to create a matrix.
// could remove at least the first array transcription.
void addTriples(int n_triples, DTriple *triples, SparseMatrix<double> *trans, SparseVector<double> *initial) {
  int r, c;
  double val;
  std::vector<T> trans_list;
  std::vector<T> initial_list;

  int max_row = 0;
  int max_col = 0;
  for (int i = 0; i < n_triples; i ++) {
    r = triples[i].row;
    c = triples[i].col;
    val = triples[i].val;

    if (r == 0) {
      initial->insert(c-1) = val;
    } else {
      trans_list.push_back(T(r-1,c-1,val));
      if (r-1 > max_row)
        max_row = r-1;
    }
    if (c-1 > max_col)
      max_col = c-1;
  }

  cerr << "max_row" << max_row << "max_col" << max_col << endl;
  assert(max_row + 1 == max_col);

  int n_states = max_col + 1;

  trans_list.push_back(T(max_row+1, max_col, 1.0));
  trans->setFromTriplets(trans_list.begin(), trans_list.end());
}

void forward_backward ( SparseVector<double> initial,
                        SparseMatrix<double> trans,
                        vector<VectorXd> emissions,
                        Map<MatrixXd> *posterior,
                        int n_states, int *permutation,
                        bool verbose) {

  if (verbose) {
    cerr << "trans(" << trans.rows() << ", " << trans.cols() << "):" << endl;
    for (int i = 0; i < trans.rows(); i++)
      cerr << " " << trans.row(i);
  }

  //int n_states = posterior->cols();
  int n_obs = emissions[0].size();
  int n_events = emissions.size();

  if (verbose) {
    cerr << "initial: " << endl << initial << endl;
  }

  cerr << "n states: " << n_states << endl;

  cerr << "n emissions[0]: " << emissions[0].size() << endl;
  cerr << "n events: " << n_events << endl;

  if (verbose) {
    cerr << "emissions: " << endl;
    for (int i = 0; i < emissions.size(); i++)
      cerr << " " << emissions[i].transpose() << endl;
  }

  //assert(emissions[0].size() == n_states);

  std::vector<SparseVector<double>> forward(n_events+1);
  forward[0] = initial;
  SparseVector<double> fromPrev;
  for (int i = 1; i <= n_events; i ++) {
    fromPrev = trans.transpose() * forward[i-1];
    for (SparseVector<double>::InnerIterator iter(fromPrev); iter; ++iter) {
      fromPrev.coeffRef(iter.index()) *= emissions[i-1][permutation[iter.index()]];
      //fromPrev.coeffRef(iter.index()) *= emissions[permutation[i-1]][iter.index()];
    }
    //SparseVector<double> withEms = fromPrev.cwiseProduct(emissions[i-1]);
    forward[i] = fromPrev / fromPrev.sum();
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
  SparseVector<double> fromNext;
  for (int i = n_events - 1; i >= 0; i --) {
    fromNext = backward[i+1];
    for (SparseVector<double>::InnerIterator iter(fromNext); iter; ++iter) {
      fromNext.coeffRef(iter.index()) *= emissions[i][permutation[iter.index()]];
    }
    //SparseVector<double> withEms = fromNext.cwiseProduct(emissions[i]);
    backward[i] = trans * fromNext;
    //backward[i] = (trans * backward[i+1].cwiseProduct(emissions[i]));
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
  for (int i = 0; i < n_events-1; i ++) {
    //SparseVector<double> row = forward[i].cwiseProduct(backward[i]);
    VectorXd row = forward[i+1].cwiseProduct(backward[i+1]);
    double s = row.sum();
    posterior->row(i) = row / s;
  }
}
