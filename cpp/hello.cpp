// 使用cin/cout必须包含iostream文件以及std命名空间
#include <iostream>

int main(){
    using namespace std;
    for (int i = 0; i < 3; i++)
    {
        /* code */
        // 尽量在首次使用变量前声明它
        int* variable;
        cout<<&variable<<endl;
        cout<<"Hello World"<<endl;
    }
    cout<<cin.get();
    return 0;
}

