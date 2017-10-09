
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <fstream>

#define FEATURES 90

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

double predict(vector<double> coef, double intercept, vector< vector<double> > val_x, vector<double> val_y){
  double r = 0;
  for(int i = 0; i < val_y.size(); i++)
    r += sq(calculate_y(val_x[i], coef, intercept) - val_y[i]);

  return sqrt(r / val_y.size());
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

void computeYearsError(vector<double> &coefs, double &intercept, vector< vector<double> > &values_x, vector<double> &values_y, vector< pair<int, double> > &years){
  years.clear();
  vector< vector<double> > small_x;
  vector<double> small_y;

  int initialYear = 1922;
  int lastYear = 2011;
  for(int i = initialYear; i <= lastYear; i++){
    small_x.clear();
    small_y.clear();

    for(int j = 0; j < values_y.size(); j++){
      if(values_y[j] == i){
        small_y.push_back(values_y[j]);
        small_x.push_back(values_x[j]);
      }
    }
    years.push_back(make_pair(i, predict(coefs, intercept, small_x, small_y)));
  }
}

int main(){
  ios::sync_with_stdio(false);

  string line;
  ifstream file;
  file.open("year-prediction-msd-test.txt");

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

  vector< pair <int, double> > years;

  // // Batch result
  // double intercept = 107.9848;
  // vector<double> coefs = {73.2742, 49.4721, 0.0000, 0.0000, 0.0000, 0.0000, 55.4656, 42.7999, 0.0000, 45.4589, 74.5383, 0.0000, 6.8444, 4.2324, 5.8435, 5.3507, 5.0135, 5.7311, 6.0581, 6.0485, 4.5193, 9.3611, 4.6458, 3.2297, 63.4223, 0.0000, 48.7292, 59.7762, 54.7448, 51.1968, 73.0185, 47.8609, 39.8976, 43.7056, 38.1531, 0.0000, 40.1372, 60.6060, 65.3707, 52.5707, 0.0000, 56.0198, 86.0092, 39.9855, 62.1996, 0.0000, 67.3722, 45.5821, 65.6287, 58.5119, 66.5308, 45.5327, 61.8614, 46.5924, 80.1981, 43.3163, 40.2274, 28.8676, 39.1276, 32.6050, 55.3887, 57.8998, 45.2567, 51.4928, 77.6156, 57.2690, 39.8822, 38.0393, 0.0000, 59.7692, 68.8672, 64.1850, 85.8442, 56.3055, 55.2990, 52.5850, 53.3976, 0.0000, 68.1402, 60.7872, 38.3904, 52.6168, 39.9634, 70.9076, 61.1768, 50.3272, 0.0000, 37.1532, 54.3720, 37.5810};

  // // Stochastic result
  // double intercept = 593.981;
  // vector<double> coefs = {52.5001, -34.9682, 0, 0, 0, 0, 30.8973, -23.8917, 0, 7.04987, 2.5173, 0, 28.0398, 32.728, -43.6041, 57.914, 45.8977, -71.3166, 34.9175, 66.7672, -10.2108, 9.09603, 33.7581, -62.5573, 9.71265, 0, 70.4291, -0.180366, 23.9274, 71.0916, 41.8813, 27.7873, 32.8633, 121.083, 128.749, 0, 10.2617, 33.1991, 97.6427, 84.5341, 0, 33.7831, 96.0464, 62.6929, 10.561, 0, 64.0049, 16.3413, 4.34439, 25.993, 42.6502, 53.2538, 103.485, 9.8059, 35.1135, 39.5688, -14.5982, 138.766, 19.4837, 56.6741, 118.48, 15.6952, 14.8014, 99.6419, 25.7225, 23.428, 6.65658, -28.9667, 0, 15.9457, 21.2792, -18.6165, 103.243, 67.8207, 62.227, 42.5644, 71.4954, 0, 146.533, 67.4901, 5.70318, 51.6954, 53.144, 19.3829, 17.4309, 54.2821, 0, 18.9829, 39.077, -5.13089};

  // GA result
  double intercept = 78.2205;
  vector<double>coefs = {57.7514, 59.3566, 48.5810, 63.7484, 86.4367, -28.6470, 42.9303, -9.3200, 58.9484, 22.7032, 76.5324, 95.1294, 4.2497, 55.7072, -38.6468, 60.1877, 50.2561, 10.4804, -9.5731, 38.1627, -9.6102, 57.2243, -31.2973, -63.4344, 56.2760, 82.1943, 70.6394, 22.7098, -73.1194, 80.8929, 24.5257, 42.7336, 66.2078, 14.4183, 73.4984, -6.4821, 58.5881, 44.1640, 9.0973, 73.8286, 60.4891, 51.6044, 2.9340, 43.3523, 87.9461, 69.3258, 84.4414, 69.0492, 76.5630, 19.5368, 60.7139, -1.2122, 34.8028, 27.3452, 75.6945, 36.6277, 45.0673, 51.6336, 75.9505, 42.3580, 15.4832, 48.0594, 32.9106, 50.3460, 94.5399, 69.4324, 96.0986, 82.1397, 2.2702, 41.6085, 64.4591, 55.2912, 85.9964, 62.5065, 45.9496, 71.8817, 78.0000, 58.4759, 73.7530, 66.1667, 90.2213, -22.9202, 83.7691, 61.8709, 74.1050, 9.4673, 38.0709, 40.9486, -43.4860, 44.1425};

  cout << predict(coefs, intercept, values_x, values_y) << endl;

  computeYearsError(coefs, intercept, values_x, values_y, years);

  for(int i = 0; i < years.size(); i++){
    cout << years[i].first << ", " << years[i].second << endl;
  }

  return 0;
}
