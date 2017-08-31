
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <iomanip>

#define FEATURES 90
#define TEST_SIZE 0.05
#define LR 1e-4
#define PR 1e-7

using namespace std;

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

int fit(vector<double> &coeff, double &intercept, vector< vector<double> > &data_x, vector<double> &data_y, double lr, double pr){

  double n = data_x[0].size();
  double m = data_y.size();

  intercept = 0;
  coeff.assign(n, 0);

  int iterations = 0;
  double i_grad;
  vector<double> c_grad;

  while(true){
    i_grad = 0;
    c_grad.assign(n, 0);

    for(int i = 0; i < m; i++){
      double y = data_y[i];
      i_grad += -(2 / m) * (y - (calculate_y(data_x[i], coeff, intercept)));

      double h = calculate_y(data_x[i], coeff, intercept);
      for(int j = 14; j < 15; j++)
        c_grad[j] += -(2 / m) * data_x[i][j] * (y - h);
      // c_grad[0] += -(2 / m) * data_x[i][0] * (y - ((coeff[0] * data_x[i][0]) + (coeff[1] * data_x[i][1]) + intercept));
      // c_grad[1] += -(2 / m) * data_x[i][1] * (y - ((coeff[0] * data_x[i][0]) + (coeff[1] * data_x[i][1]) + intercept));
    }
    intercept = intercept - (lr * i_grad);
    for(int j = 14; j < 15; j++)
      coeff[j] = coeff[j] - (lr * c_grad[j]);
    // coeff[0] = coeff[0] - (lr * c_grad[0]);
    // coeff[1] = coeff[1] - (lr * c_grad[1]);
    // cout << intercept << " " << coeff[0] << " " << coeff[1] << endl;

    iterations++;
    if(iterations >= 100)
      break;

  }

  return iterations;
}

void train_test_split(vector< vector<double> > &values_x, vector<double> &values_y, vector< vector<double> > &val_x, vector< vector<double> > &train_x, vector<double> &val_y, vector<double> &train_y, double r){
  int limit = (1 - r) * values_x.size();
  int size = values_x.size();

  val_x.insert(val_x.begin(), values_x.begin(), values_x.begin() + limit);
  val_y.insert(val_y.begin(), values_y.begin(), values_y.begin() + limit);

  train_x.insert(train_x.begin(), values_x.begin() + limit, values_x.end());
  train_y.insert(train_y.begin(), values_y.begin() + limit, values_y.end());

}

double sq(double a){
  return a * a;
}

double predict(vector<double> coef, double intercept, vector< vector<double> > val_x, vector<double> val_y){
  double r = 0;
  for(int i = 0; i < val_y.size(); i++){
    r += sq(calculate_y(val_x[i], coef, intercept) - val_y[i]);

    cout << calculate_y(val_x[i], coef, intercept) << " " << val_y[i] << endl;
  }


  return sqrt(r / val_y.size());
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

  vector< vector<double> > train_x, val_x;
  vector<double> train_y, val_y;

  train_test_split(values_x, values_y, val_x, train_x, val_y, train_y, TEST_SIZE);

  vector<double> coef;
  double intercept;
  cout << "Starting train" << endl;
  fit(coef, intercept, train_x, train_y, LR, PR);
  cout << "Done with training" << endl;

  cout << predict(coef, intercept, val_x, val_y) << endl;

  return 0;
}
