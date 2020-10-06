#include<iostream>
#include<vector>
#include<string.h>

using namespace std;

int main()
{
    int test, check;

    cin>>test>>check;

    vector < string > store(test);

    for(int i=0; i<(test+check); i++)
    {
        string temp;
        cin>>temp;

        store.push_back(temp);
    }

    cout<<endl;

    for(int i=0; i<(test+check); i++)
        cout<<store[i]<<"\n";

    return 0;
}
