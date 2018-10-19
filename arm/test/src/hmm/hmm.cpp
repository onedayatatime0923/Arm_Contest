
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
  data.push_back("you1");
  data.push_back("ing");
  data.push_back("do");
  data.push_back("what");
  data.push_back("I");
  data.push_back("say");
  data.push_back("one");
  data.push_back("if");
  data.push_back("you2");
  data.push_back("feel1");
  data.push_back("alone");
  data.push_back("can");
  data.push_back("watch");
  data.push_back("ghost");
  data.push_back("movie");
  data.push_back("you3");
  data.push_back("will");
  data.push_back("feel2");
  data.push_back("restroom");
  data.push_back("have1");
  data.push_back("people1");
  data.push_back("bed");
  data.push_back("down");
  data.push_back("have2");
  data.push_back("people2");
  data.push_back("kitchen");
  data.push_back("have3");
  data.push_back("people3");
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
