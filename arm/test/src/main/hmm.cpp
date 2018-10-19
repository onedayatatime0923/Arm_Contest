
#include <iostream>
#include "../hmm/hmm.h"

using namespace std;

int main()
{
  Hmm hmm;

	cout << hmm.query("judge", "good")<< endl;
	cout << hmm.query("system", "sign")<< endl;
		
	return 0;
}
