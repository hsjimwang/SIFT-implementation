#include <iostream>

using namespace std;

int main(){
    int num = 1;
    if(false & num++ & num++){
        cout << "A" << endl;
    }
    cout << "num = " << num << endl;
}