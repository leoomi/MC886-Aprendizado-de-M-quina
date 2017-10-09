
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <cfloat>

#define FEATURES 90
#define TEST_SIZE 0.05
#define NUM_ITERATIONS 100
#define LR 1e-2
#define PR 1e-7

using namespace std;

double sq(double a){
  return a * a;
}

double calculate_y(vector<double> x, vector<double> coeff, double intercept){
  double acc = 0;
  for(int i = 0; i < x.size(); i++)
    acc += x[i] * coeff[i];

  return acc + intercept;
}

vector<double> product(vector<double> a, vector<double> b){
  if(a.size() != b.size()){
    cout << "Vectors should have the same size" << endl;
    exit(1);
  }

  vector<double> r;
  for(int i = 0; i < a.size(); i++)
    r.push_back(a[i] * b[i]);

  return r;
}

vector<double> product(vector<double> a, vector< vector<double> > b, int p){
  if(a.size() != b.size()){
    cout << "Vectors should have the same size" << endl;
    exit(1);
  }

  vector<double> r;
  for(int i = 0; i < a.size(); i++)
    r.push_back(a[i] * b[i][p]);

  return r;
}

double sum(vector<double> a){
  double acc = 0;
  for(int i = 0; i < a.size(); i++)
    acc += a[i];
  return acc;
}

vector<double> subV(vector<double> a, vector<double> b){
  if(a.size() != b.size()){
    cout << "Vectors should have the same size" << endl;
    exit(1);
  }

  vector<double> r;
  for(int i = 0; i < a.size(); i++)
    r.push_back(a[i] - b[i]);

  return r;
}

int fit(vector<double> &coeff, double &intercept, vector< vector<double> > &data_x, vector<double> &data_y, double lr, double pr, int num_it, vector<bool> weights){

  double n = data_x[0].size();
  double m = data_y.size();

  intercept = 0;
  coeff.assign(n, 0);

  int iterations = 0;
  double i_grad;
  vector<double> c_grad;

  while(num_it--){
    i_grad = 0;
    c_grad.assign(n, 0);

    for(int i = 0; i < m; i++){
      double y = data_y[i];
      i_grad += -(2 / m) * (y - (calculate_y(data_x[i], coeff, intercept)));

      double h = calculate_y(data_x[i], coeff, intercept);
      for(int j = 0; j < n; j++)
        if(weights[j])
          c_grad[j] += -(2 / m) * data_x[i][j] * (y - h);
    }

    intercept = intercept - (lr * i_grad);
    for(int j = 0; j < n; j++)
      if(weights[j])
        coeff[j] = coeff[j] - (lr * c_grad[j]);

    iterations++;
  }

  return iterations;
}

void normalize(vector< vector<double> > &data){
  vector<double> min, max;
  min.assign(data.front().size(), DBL_MAX);
  max.assign(data.front().size(), -DBL_MAX);
  for(int i = 0; i < data.size(); i++){
    for(int j = 0; j < data.front().size(); j++){
      min[j] = (min[j] < data[i][j] ? min[j] : data[i][j]);
      max[j] = (max[j] > data[i][j] ? max[j] : data[i][j]);
    }
  }

  for(int i = 0; i < data.size(); i++)
    for(int j = 0; j < data.front().size(); j++)
      data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
}

void train_test_split(vector< vector<double> > &values_x, vector<double> &values_y, vector< vector<double> > &val_x, vector< vector<double> > &train_x, vector<double> &val_y, vector<double> &train_y, double r){
  int limit = (1 - r) * values_x.size();
  int size = values_x.size();

  val_x.insert(val_x.begin(), values_x.begin(), values_x.begin() + limit);
  val_y.insert(val_y.begin(), values_y.begin(), values_y.begin() + limit);

  train_x.insert(train_x.begin(), values_x.begin() + limit, values_x.end());
  train_y.insert(train_y.begin(), values_y.begin() + limit, values_y.end());

}

double predict(vector<double> coef, double intercept, vector< vector<double> > val_x, vector<double> val_y){
  double r = 0;
  for(int i = 0; i < val_y.size(); i++)
    r += sq(calculate_y(val_x[i], coef, intercept) - val_y[i]);

  return sqrt(r / val_y.size());
}

double score(vector<double> coef, double intercept, vector< vector<double> > val_x, vector<double> val_y){
  double count = 0;

  for(int i = 0; i < val_y.size(); i++)
    if((int)(calculate_y(val_x[i], coef, intercept)) == (int)(val_y[i]))
      count++;

  return count / val_y.size();
}

int main(){
  ios::sync_with_stdio(false);

  string line;
  ifstream file;
  file.open("year-prediction-msd-train.txt");

  if(!file.is_open()){
    cout << "Error when opening the file" << endl;
    return 1;
  }

  vector< vector<double> > values_x;
  vector<double> values_y;

  double f, year;
  char c;

  cout << "Loading data" << endl;
  while(file.peek() != EOF){
    file >> year;
    values_y.push_back(year);

    values_x.push_back(vector<double>());
    values_x.back().assign(FEATURES, 0);

    for(int i = 0; i < FEATURES; i++)
      file >> c >> values_x.back()[i];
  }
  values_x.pop_back();
  values_y.pop_back();

  normalize(values_x);

  vector< vector<double> > train_x, val_x;
  vector<double> train_y, val_y;

  train_test_split(values_x, values_y, val_x, train_x, val_y, train_y, TEST_SIZE);

  vector<double> coef;
  double intercept;

  vector<bool>weights;
  weights.assign(FEATURES, true);
  weights[11] = false;
  weights[8] = false;
  weights[86] = false;
  weights[35] = false;
  weights[5] = false;
  weights[2] = false;
  weights[3] = false;
  weights[77] = false;
  weights[25] = false;
  weights[40] = false;
  weights[68] = false;
  weights[45] = false;
  weights[4] = false;

  fit(coef, intercept, train_x, train_y, LR, PR, NUM_ITERATIONS, weights);

  cout << fixed << setprecision(4) << predict(coef, intercept, val_x, val_y) << endl;

  for(int i = 0; i < coef.size(); i++)
    cout << coef[i] << " ";
  cout << endl;
  cout << intercept << endl;

  return 0;
}
