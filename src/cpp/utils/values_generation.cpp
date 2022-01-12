#include <tbsla/cpp/utils/values_generation.hpp>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>

std::tuple<std::size_t, std::size_t, double, std::size_t> tbsla::utils::values_generation::cdiag_value(std::size_t i, std::size_t nv, std::size_t nr, std::size_t nc, std::size_t cdiag) {
  if(cdiag == 0) {
    return std::make_tuple(i, i, 1, 10);
  }
  if(i < std::max(std::min((long int)nc - (long int)cdiag, (long int)cdiag), (long int)0)) {
    return std::make_tuple(i, i + cdiag, 1, 30);
  } else if (i < std::max((long int)cdiag + 2 * ((long int)nc - 2 * (long int)cdiag), (long int)0)) {
    long int it = (i + cdiag) / 2;
    if(i % 2 == 0) {
      return std::make_tuple(it, it - cdiag, 1, 31);
    } else {
      return std::make_tuple(it, it + cdiag, 1, 32);
    }
  } else {
    long int it = i - ((long int)nc - 2 * (long int)cdiag);
    if(cdiag > nc) {
      it -= cdiag - nc;
    }
    return std::make_tuple(it, it - cdiag, 1, 33);
  }
}

std::tuple<std::size_t, std::size_t, double, std::size_t> tbsla::utils::values_generation::cqmat_value(std::size_t i, std::size_t nr, std::size_t nc, std::size_t c_, double q, unsigned int seed_mult) {
  unsigned int seedp = i;
  if(seed_mult > 0) {
    seedp = seed_mult * i;
  }
  std::size_t c = std::min(nc, c_);
  if(i < c * std::min(nc - c + 1, nr)) {
    return std::make_tuple(i / c, ((double)rand_r(&seedp) / RAND_MAX) < q ? rand_r(&seedp) % nc : i / c + i % c, 1, 30);
  } else {
    std::size_t n_full_rows = std::min(nc - c + 1, nr);
    std::size_t it = i - c * std::min(nc - c + 1, nr);
    std::size_t curr_row = 1;
    while(curr_row < nr - n_full_rows && it >= c - curr_row) {
      it -= (c - curr_row);
      curr_row++;
    }
    curr_row += n_full_rows - 1;
    return std::make_tuple(curr_row, (double)(rand_r(&seedp) / RAND_MAX) < q ? rand_r(&seedp) % nc : curr_row + it, 1, 31);
  }
}

double* tbsla::utils::values_generation::cqmat_sum_columns(std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned seed_mult) {
  std::size_t gnv = 0;
  for(std::size_t i = 0; i < std::min(nc - std::min(c, nc) + 1, nr); i++) {
    gnv += std::min(c, nc);
  }
  for(std::size_t i = 0; i < std::min(nr, nc) - std::min(nc - std::min(c, nc) + 1, nr); i++) {
    gnv += std::min(c, nc) - i - 1;
  }

  double* sum = new double[nc];
  for (int i = 0; i < nc; i++) {
    sum[i] = 0;
  }
  for(std::size_t i = 0; i < gnv; i++) {
    auto tuple = tbsla::utils::values_generation::cqmat_value(i, nr, nc, c, q, seed_mult);
    sum[std::get<1>(tuple)] += std::get<2>(tuple);
  }
  return sum;
}

// Loop to ensure we get the specified number of columns required
// i.e. to compensate for duplicates in random generation
// More efficient than shuffling all columns, in (expected) cases with large n and relatively small nnz
// DO NOT USE AS ALERNATIVE TO SHUFFLING IF YOU WANT > 50% OF VALUES IN THE RANGE OR SOMETHING
/*int* tbsla::utils::values_generation::random_columns(std::size_t n_vals, std::size_t range, std::uniform_real_distribution<double> distr_ind, std::default_random_engine generator) {
  int* cols;
  if(n_vals>0) {
	cols = new int[n_vals];
	bool* is_used = new bool[range];
	//int n_missing;
	//while((n_missing = n_vals - tmp.size()) > 0) {
	//std::vector<int> tmp;
	for(int i=0; i<n_vals; i++) {
		//int col_ind = (int)(distr_ind(generator));
		int col_ind = rand() % range;
		while(is_used[col_ind])
			//col_ind = (int)(distr_ind(generator));
			col_ind = rand() % range;
		cols[i] = col_ind;
		is_used[col_ind] = true;
	}
	delete[] is_used;
  }
  return cols;
}*/

/*int* tbsla::utils::values_generation::random_columns(std::size_t i, std::size_t n_vals, std::size_t nc, unsigned seed_mult) {
  int* cols;
  unsigned int seedp = i;
  if(seed_mult > 0) {
    seedp = seed_mult * i;
  }
  if(n_vals>0) {
    cols = new int[n_vals];
    bool* is_used = new bool[nc];
    for(int k=0; k<n_vals; k++) {
      int col_ind = rand_r(&seedp) % nc;
      while(is_used[col_ind])
	col_ind = rand_r(&seedp) % nc;
      cols[k] = col_ind;
      is_used[col_ind] = true;
    }
    delete[] is_used;
  }
  return cols;
}*/


/*int* tbsla::utils::values_generation::fix_list(int* list, std::size_t n_vals, std::size_t nc) {
  int* fixed_list = new int[n_vals];
  bool* is_used = new bool[nc];
  for(int z=0; z<nc; z++)
    is_used[z] = false;
  bool* to_fix = new bool[n_vals];
  for(int z=0; z<n_vals; z++)
    to_fix[z] = false;
  for(int k=0; k<n_vals; k++) {
    int col_ind = list[k];
    if(is_used[col_ind]) {
      to_fix[k] = true;
      //std::cout << col_ind << " already used ; need to fix element " << k << std::endl;
    }
    is_used[col_ind] = true;
    //std::cout << col_ind << " is used\n";
  }
  for(int k=0; k<n_vals; k++) {
      if(to_fix[k]) {
	int new_col = list[k];
	int mod = 1;
	if(new_col>(nc/2))
	  mod = -1;
	while(is_used[new_col] && new_col > 0 && new_col < nc-1) {
	  //std::cout << new_col << " is used ; moving to ";
	  new_col += mod;
	  //std::cout << new_col << std::endl;
	}
	if(!is_used[new_col]) {
	  fixed_list[k] = new_col;
	  is_used[new_col] = true;
	}
	else {
	  //std::cout << "Ran out of values to shift, in 'fix_list'\n";
	}
      }
      else {
	fixed_list[k] = list[k];
      }
  }
  delete[] is_used;
  delete[] to_fix;
  return fixed_list;
}*/

int* tbsla::utils::values_generation::fix_list(int* list, std::size_t n_vals, std::size_t nc) {
  int* fixed_list = new int[n_vals];
  std::unordered_map<int,bool> is_used;
  bool* to_fix = new bool[n_vals];
  for(int z=0; z<n_vals; z++)
    to_fix[z] = false;
  for(int k=0; k<n_vals; k++) {
    int col_ind = list[k];
    if(is_used.find(col_ind) != is_used.end()) {
      to_fix[k] = true;
      //std::cout << col_ind << " already used ; need to fix element " << k << std::endl;
    }
    is_used[col_ind] = true;
    //std::cout << col_ind << " is used\n";
  }
  for(int k=0; k<n_vals; k++) {
      if(to_fix[k]) {
	int new_col = list[k];
	int mod = 1;
	if(new_col>(nc/2))
	  mod = -1;
	while(is_used.find(new_col) != is_used.end() && new_col > 0 && new_col < nc-1) {
	  //std::cout << new_col << " is used ; moving to ";
	  new_col += mod;
	  //std::cout << new_col << std::endl;
	}
	if(is_used.find(new_col) == is_used.end()) {
	  fixed_list[k] = new_col;
	  is_used[new_col] = true;
	}
	else {
	  //std::cout << "Ran out of values to shift, in 'fix_list'\n";
	}
      }
      else {
	fixed_list[k] = list[k];
      }
  }
  is_used.clear();
  delete[] to_fix;
  return fixed_list;
}


int* tbsla::utils::values_generation::random_columns(std::size_t i, std::size_t n_vals, std::size_t nc, unsigned seed_mult) {
  int* cols;
  unsigned int seedp = i;
  if(seed_mult > 0) {
    seedp = seed_mult * i;
  }
  if(n_vals>0) {
    int* raw_cols = new int[n_vals];
    for(int k=0; k<n_vals; k++) {
      int col_ind = rand_r(&seedp) % nc;
      raw_cols[k] = col_ind;
    }
    cols = tbsla::utils::values_generation::fix_list(raw_cols, n_vals, nc);
  }
  return cols;
}



