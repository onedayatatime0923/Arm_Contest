
#include <string>
#include "hmm.h"

using namespace std;

Edge::Edge(const string& one, const string& two, const float& value):
  _one(one), _two(two), _value(value){};

const float& Edge::getValue() const{
  return _value;
}

Hmm::Hmm(const float& _default): _default(_default){
  _edge.push_back(Edge("dummy", "judge", 80));
  _edge.push_back(Edge("judge", "good" , 80));
  _edge.push_back(Edge("good" , "we"   , 80));
  _edge.push_back(Edge("we"   , "is"   , 80));
  _edge.push_back(Edge("is"   , "ten"  , 80));
  _edge.push_back(Edge("ten"  , "group", 80));
  _edge.push_back(Edge("group", "we"   , 80));
  _edge.push_back(Edge("we"   , "great", 80));
  _edge.push_back(Edge("great", "make", 80));
  _edge.push_back(Edge("make", "sign", 80));
  _edge.push_back(Edge("sign", "translate", 80));
  _edge.push_back(Edge("translate", "system", 80));
  _edge.push_back(Edge("system", "help", 80));
  _edge.push_back(Edge("help", "sign", 80));
  _edge.push_back(Edge("sign", "human", 80));
  _edge.push_back(Edge("human", "increase", 80));
  _edge.push_back(Edge("increase", "job", 80));
  _edge.push_back(Edge("job", "chance", 80));
  _edge.push_back(Edge("chance", "change", 80));
  _edge.push_back(Edge("change", "they", 80));
  _edge.push_back(Edge("they", "of", 80));
  _edge.push_back(Edge("of", "life", 80));
}

const float& Hmm::query(const string& one, const string& two) const{
  for(int i = 0;i < _edge.size();++i){
    if(_edge[i](one, two)){
      return _edge[i].getValue();
    }
  }
  return _default;
}
