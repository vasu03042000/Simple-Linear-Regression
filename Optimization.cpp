// Optimizing the GCC compiler to the highest possible level ~o3 optimization
// Refer to the link for more detail - https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
#pragma GCC optimize(3)

#include <bits/stdc++.h>

using namespace std;

#define endl "\n"

// Defining multi-line preprocessor macros
// For fast I/P O/P operations, {cin==scanf()}
// Method 1
#define fast {                                  \
                ios::sync_with_stdio(false);    \
                cin.tie(NULL);                  \
                cout.tie(NULL);                 \
             }

/* OR You can define this macro as
#define fast                        \
        ios:sync_with_stdio(false); \
        cin.tie(NULL);              \
        cout.tie(NULL/0)
*/

int main()
{
    // To run the set of statements
    // Defined as a macro above
    fast;
    cout<<1199999997%1000000007;
    return 0;
}

