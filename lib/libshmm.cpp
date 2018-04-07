#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
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
                        Map<MatrixXd> *posteriorFull,
                        Map<VectorXd> *posteriorSummed,
                        int n_events, int *permutation_,
                        bool verbose);

struct DTriple {
  int row;
  int col;
  double val;
};

void addTriples(int n_triples, DTriple *triples, SparseMatrix<double> *trans, SparseVector<double> *initial);

bool verbose = false;

int shmm(int n_triples, DTriple *triples,
         int n_states, int n_obs, double **emissions_aptr,
         int n_events, int *permutation,
         bool sum,
         double *posterior_arr ) {
  if (verbose) {
    cerr << "params: " << "n_triples=" << n_triples << ", n_states=" << n_states << ", n_obs=" << n_obs << ", n_events=" << n_events << endl;

    cerr.precision(17);
    double triple_val_hash = 0.0;
    for (int i = 0; i < n_triples; i++) {
      triple_val_hash += triples[i].val;
    }
    cerr << "triples hash: " << triple_val_hash << endl;

    double permutation_hash = 0.0;
    for (int i = 0; i < n_states; i++) {
      permutation_hash += permutation[i];
    }
    cerr << "permutation hash: " << permutation_hash << endl;

    double emissions_hash = 0.0;
    for (int r = 0; r < n_events; r++) {
      for (int c = 0; c < n_obs+1; c++) {
        emissions_hash += emissions_aptr[r][c];
      }
    }
    cerr << "emissions hash: " << emissions_hash << endl;
  }
  SparseMatrix<double> trans(n_states+1, n_states+1);
  SparseVector<double> initial(n_states+1);
  addTriples(n_triples, triples, &trans, &initial);

  if (verbose) {
    cerr << "trans(" << trans.rows() << ", " << trans.cols() << "):" << endl;
    for (int i = 0; i < trans.rows(); i++)
      cerr << " " << trans.row(i);
    cerr << "init: " << initial;
    cerr << "permutation: " << permutation[0] << permutation[1] << permutation[2] << endl;
  }

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

  if (sum) {
    for (int i = 0; i < n_states + 1; i++) {
      posterior_arr[i] = 0;
    }
    Map<VectorXd> posteriorSummed(posterior_arr, n_states + 1);
    forward_backward ( initial, trans, emissions, NULL, &posteriorSummed, n_states+1, permutation, verbose );
    if (verbose)
      cerr << "posterior: " << endl << posteriorSummed << endl;
  } else {
    Map<MatrixXd> posteriorFull(posterior_arr, n_events, n_states + 1);
    forward_backward ( initial, trans, emissions, &posteriorFull, NULL, n_states+1, permutation, verbose );
    if (verbose)
      cerr << "posterior: " << endl << posteriorFull << endl;
  }
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

  trans_list.push_back(T(max_row+1, max_col, 1.0));
  trans->setFromTriplets(trans_list.begin(), trans_list.end());
}

void forward_backward ( SparseVector<double> initial,
                        SparseMatrix<double> trans,
                        vector<VectorXd> emissions,
                        Map<MatrixXd> *posteriorFull,
                        Map<VectorXd> *posteriorSummed,
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
    }
    forward[i] = fromPrev / fromPrev.sum();
  }

  if (verbose) {
    cerr << "forward: " << endl;
    for (int i = 0; i < forward.size(); i++)
      cerr << " " << forward[i];
  }

  VectorXd row;
  SparseVector<double> backward = MatrixXd::Ones(n_states,1).col(0).sparseView();
  for (int i = n_events - 1; i > 0; i --) {
    for (SparseVector<double>::InnerIterator iter(backward); iter; ++iter) {
      backward.coeffRef(iter.index()) *= emissions[i][permutation[iter.index()]];
    }
    backward = trans * backward;
    backward /= backward.sum();

    row = forward[i].cwiseProduct(backward);
    if (posteriorFull != NULL) {
      posteriorFull->row(i-1) = row / row.sum();
    }
    else if (posteriorSummed != NULL) {
      *posteriorSummed += row / row.sum();
    }
  }

  // cerr << "summed result: " << *posteriorSummed << endl;

  // use -1 to avoid reporting the end token
  //for (int i = 1; i < n_events; i ++) {
    //SparseVector<double> row = forward[i].cwiseProduct(backward[i]);
  //}
}
