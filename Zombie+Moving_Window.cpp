#include<iostream>
#include<vector>

using namespace std;

// MOVING WINDOW ALGORITHM
int max_window(vector <int> arr, int n, int k)
{
    int sum = 0;
    int cnt = 0, maxcnt = 0;

    for (int i = 0; i < n; i++)
    {
        // If adding current element doesn't
        // cross limit add it to current sub-array
        if ((sum + arr[i]) > k)
        {
            sum += arr[i];
            cnt++;
        }

        // Else, remove first element of current
        // sub-array and add the current element
        else if(sum!=0)
        {
            sum = sum - arr[i - cnt] + arr[i];
        }

        maxcnt = max(cnt, maxcnt);
    }

    return maxcnt;
}

int max_sub(vector <int> arr, int n, int x)
{
    // Making a -1/1 array from the original
    // If sum of the sub-array>0 then we can pick it
    for(int i=0; i<n; i++)
    {
        if(arr[i]<=x)
            arr[i]=-1;

        else
            arr[i]=1;
    }

    return max_window(arr, n, 0);
}

int main()
{
    int n, x;
    cin>>n;

    vector <int> E(n);
    for(int i=0; i<n; i++)
        cin>>E[i];

    cin>>x;

    cout<<max_sub(E,n,x);

    return 0;
}

