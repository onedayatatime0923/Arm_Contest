
#include <string>
#include <vector>

using namespace std;

class Edge{
  public:
    Edge(const string& one, const string& two, const float& value);

    inline bool operator() (const string& one, const string& two) const;

    const float& getValue() const;

  private:
    string _one;
    string _two;
    float  _value;
};
inline bool Edge::operator() (const string& one, const string& two) const{
  return (this->_one == one && this->_two == two);
}

class Hmm{
  public:
    Hmm(const float& _default = 0);

    void construct(vector<string>& data);

    const float& query(const string& one, const string& two) const;
    
  private:
    float _default;
    vector<Edge> _edge;
};
