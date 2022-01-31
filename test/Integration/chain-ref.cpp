#include <iostream>
#include <vector>

using namespace std;

void fill(vector<vector<int>> &A, int fillCst) {
  int rows = A.size();
  int cols = A[0].size();
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      A[i][j] = fillCst;
}

void print(vector<vector<int>> &A) {
  int rows = A.size();
  int cols = A[0].size();
  cout << "( ";
  for (int i = 0; i < rows; i++) {
    cout << "( ";
    for (int j = 0; j < cols; j++) {
      if (j == cols - 1)
        cout << A[i][j] << " ";
      else
        cout << A[i][j] << ", ";
    }
    if (i == rows - 1)
      cout << ") ";
    else
      cout << "), ";
  }
  cout << ")\n";
}

vector<vector<int>> gemm(vector<vector<int>> &A, vector<vector<int>> &B) {
  int rows = A.size();
  int cols = B[0].size();
  vector<vector<int>> C(rows, vector<int>(cols, 0));

  int I = C.size();
  int J = C[0].size();
  int K = B.size();
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++)
        C[i][j] += A[i][k] * B[k][j];
  return C;
}

int main() {
  vector<vector<int>> A1(30, vector<int>(35, 0));
  vector<vector<int>> A2(35, vector<int>(15, 0));
  vector<vector<int>> A3(15, vector<int>(5, 0));
  vector<vector<int>> A4(5, vector<int>(10, 0));
  vector<vector<int>> A5(10, vector<int>(20, 0));
  vector<vector<int>> A6(20, vector<int>(25, 0));

  fill(A1, 1);
  fill(A2, 2);
  fill(A3, 3);
  fill(A4, 4);
  fill(A5, 5);
  fill(A6, 6);

  auto I1 = gemm(A1, A2);
  auto I2 = gemm(I1, A3);
  auto I3 = gemm(I2, A4);
  auto I4 = gemm(I3, A5);
  auto I5 = gemm(I4, A6);

  print(I5);

  return 0;
}
