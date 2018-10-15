
#include <iostream>
#include "../hmm/hmm.h"

using namespace std;

int main()
{
  Hmm hmm;

	cout << hmm.query("good", "judge");
	cout << hmm.query("system", "sign");
		
	return 0;
}
