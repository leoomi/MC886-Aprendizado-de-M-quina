
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <cfloat>
#include <ctime>
#include <cstdlib>

#define FEATURES 90
#define TEST_SIZE 0.05
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

class Chromosome{
public:
  vector<double> coefs;
  double intercept;
  double eval;
  Chromosome(vector<double> coefs, double intercept) : coefs(coefs), intercept(intercept), eval(-1){
  }

  double evaluate(vector< vector<double> > x, vector<double> y){
    if(eval != -1)
      return eval;

    int pos = rand() % x.size();

    eval = predict(coefs, intercept, x, y);

    return eval;
  }

  Chromosome crossover(Chromosome c){
    Chromosome r(coefs, intercept);

    for(int i = 0; i < coefs.size(); i++){
      double b = ((double)(rand() % 100)) / 100.0;
      r.coefs[i] = b * c.coefs[i] + (1 - b) * r.coefs[i];
    }


    double b = ((double)(rand() % 100)) / 100.0;
    r.intercept = b * c.intercept + (1 - b) * r.intercept;

    return r;
  }

  Chromosome mutate(){
    if(rand() % 100 < 20){
      int a = rand() % coefs.size();
      int b = rand() % coefs.size();
      if(b < a)
        swap(a, b);

      for(int i = a; i <= b; i++)
        coefs[i] = rand() % 201 - 100;

      if(rand() % 100 < 50)
        intercept = rand() % 201 - 100;
    }
    return *this;
  }

};

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

  srand(0);

  // build
  cout << "Starting GA" << endl;
  time_t start, curr;
  time(&start);
  vector< Chromosome > population, newPop;
  int pSize = 25;
  for(int i = 0; i < pSize; i++){
    vector<double> coef;
    double intercept;
    for(int i = 0; i < FEATURES; i++)
      coef.push_back(rand() % 201 - 100);
    intercept = rand() % 201 - 100;

    population.push_back(Chromosome(coef, intercept));
  }

  int k = 0;
  while(true){
    for(int i = 0; i < pSize; i++){
      if(population[i].evaluate(train_x, train_y) < population[0].evaluate(train_x, train_y))
        swap(population[i], population[0]);
    }
    newPop.clear();
    newPop.push_back(population.front());

    cout << fixed << setprecision(4) << k++ << ", " << population.front().evaluate(train_x, train_y) << ": ";
    for(int i = 0; i < population.front().coefs.size(); i++)
      cout << population.front().coefs[i] << " ";
    cout << ": " << population.front().intercept << endl;
    time(&curr);
    cout << "elapsed time: " << difftime(curr, start) << "s" << endl << endl;

    while(newPop.size() < pSize){
      // chose parents
      int a, b, t;
      a = rand() % pSize;
      b = rand() % pSize;
      for(int i = 0; i < 3; i++){
        do{
          t = rand() % pSize;
        }while(t == a);

        if(population[a].evaluate(train_x, train_y) > population[t].evaluate(train_x, train_y))
          a = t;
      }

      for(int i = 0; i < 3; i++){
        do{
          t = rand() % pSize;
        }while(t == b || t == a);

        if(population[b].evaluate(train_x, train_y) > population[t].evaluate(train_x, train_y))
          b = t;
      }

      newPop.push_back(population[a].crossover(population[b]).mutate());
    }

    population = newPop;
  }

  return 0;
}
