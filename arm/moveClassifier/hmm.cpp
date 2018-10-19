
#include <string>
#include "hmm.h"

using namespace std;

Edge::Edge(const string& one, const string& two, const float& value):
  _one(one), _two(two), _value(value){};

const float& Edge::getValue() const{
  return _value;
}

Hmm::Hmm(const float& _default): _default(_default){
  vector<string> data;
  data.push_back("judge");
  data.push_back("good");
  data.push_back("we");
  this->construct(data);
}

void Hmm::construct(vector<string>& data){
  string lastState = "dummy";
  for(int i = 0;i < data.size();++i){
    _edge.push_back(Edge(lastState, data[i], 80));
    lastState = data[i];
  }
}


const float& Hmm::query(const string& one, const string& two) const{
  for(int i = 0;i < _edge.size();++i){
    if(_edge[i](one, two)){
      return _edge[i].getValue();
    }
  }
  return _default;
}
