from model import Model
from cffi import FFI
import re

ctwnode_h = """
typedef struct _ctwnode_t {
  double base_counts[ALPHABET_SIZE];
  double base_log_prob;
  double children_log_prob;
  double log_prob;
  struct _ctwnode_t *children[ALPHABET_SIZE];
  unsigned int _refcount;
} ctwnode_t;

double ctwnode_update(ctwnode_t *node, char symbol, double weight, char *context, int ctxtlen);
double ctwnode_log_predict(ctwnode_t *node, char symbol, char *context, int ctxtlen);
ctwnode_t *ctwnode_new();
ctwnode_t *ctwnode_copy(ctwnode_t *self);
void ctwnode_free(ctwnode_t *self);
int ctwnode_size(ctwnode_t *self);
"""

ctwnode_c = """
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct _ctwnode_t {
  double base_counts[ALPHABET_SIZE];
  double base_log_prob;
  double children_log_prob;
  double log_prob;
  struct _ctwnode_t *children[ALPHABET_SIZE];
  unsigned int _refcount;
} ctwnode_t;

double logsumexp2(double a, double b)
{
  if (a > b) {
    return log(1.0 + exp(b - a)) + a;
  } else {
    return log(1.0 + exp(a - b)) + b;
  }
}

double logsumexp(double *a, unsigned int n)
{
  double shift = a[0];
  double rv = 0;

  for(int i=1; i<n; i++) {
    if (a[i] > shift) shift = a[i];
  }

  for(int i=0; i<n; i++) {
    rv += exp(a[i] - shift);
  }

  return log(rv) + shift;
}

ctwnode_t *ctwnode_new()
{
  ctwnode_t *self = malloc(sizeof(ctwnode_t));
  for(int i=0; i<ALPHABET_SIZE; i++) self->base_counts[i] = 1.0 / ALPHABET_SIZE;
  self->base_log_prob = self->children_log_prob = self->log_prob = 0.0;
  for(int i=0; i<ALPHABET_SIZE; i++) self->children[i] = NULL;
  self->_refcount = 1;
  return self;
}

void ctwnode_free(ctwnode_t *self)
{
  self->_refcount--;
  if (self->_refcount == 0) {
    for(int i=0; i<ALPHABET_SIZE; i++) {
      if (self->children[i]) ctwnode_free(self->children[i]);
    }
    free(self);
  }
}

ctwnode_t *ctwnode_copy(ctwnode_t *self)
{
  ctwnode_t *copy = malloc(sizeof(ctwnode_t));
  (*copy) = (*self);
  copy->_refcount = 1;
  for(int i=0; i<ALPHABET_SIZE; i++) {
    if (self->children[i]) self->children[i]->_refcount++;
  }
  return copy;
}
  
void ctwnode_base_update(ctwnode_t *self, char symbol, double weight)
{
  double sum_counts = 0.0;
  for(int i=0; i<ALPHABET_SIZE; i++) sum_counts += self->base_counts[i];
  self->base_log_prob += log(self->base_counts[symbol] / sum_counts);
  self->base_counts[symbol] += weight;
}

double ctwnode_update(ctwnode_t *self, char symbol, double weight, char *context, unsigned int ctxtlen)
{
  double orig_log_prob = self->log_prob;

  ctwnode_base_update(self, symbol, weight);

  if (ctxtlen) {
    int cnext = context[ctxtlen-1];
    ctwnode_t *child = self->children[cnext];
    if (!child) child = self->children[cnext] = ctwnode_new();
    else if (child->_refcount > 1) {
      child->_refcount--;
      child = self->children[cnext] = ctwnode_copy(child);
    }

    self->children_log_prob += ctwnode_update(child, symbol, weight, context, ctxtlen-1);
    self->log_prob = log(0.5) + logsumexp2(self->base_log_prob, self->children_log_prob);
  } else {
    self->log_prob = self->base_log_prob;
  }

  return self->log_prob - orig_log_prob;
}

double ctwnode_log_predict(ctwnode_t *self, char symbol, char *context, unsigned int ctxtlen)
{
  double sum_counts = 0;

  for(int i=0; i<ALPHABET_SIZE; i++) sum_counts += self->base_counts[i];

  double base_log_prob = self->base_log_prob + log(self->base_counts[symbol] / sum_counts);

  if (ctxtlen) {
    int cnext = context[ctxtlen-1];
    ctwnode_t *child =self->children[cnext];
    if (!child) child = self->children[cnext] = ctwnode_new();

    double children_log_prob = self->children_log_prob + ctwnode_log_predict(child, symbol, context, ctxtlen-1);

    return log(0.5) + logsumexp2(base_log_prob, children_log_prob) - self->log_prob;
  } else {
    return base_log_prob - self->log_prob;
  }
}

int ctwnode_size(ctwnode_t *self)
{
  int size = 1;
  for(int i=0; i<ALPHABET_SIZE; i++) {
    if (self->children[i]) size += ctwnode_size(self->children[i]);
  }
  return size;
}
"""

class CTW_KT(Model):
    """Context Tree Weighting over KT models with a binary alphabet.

    A specialized memory and time efficient version of model.CTW with
    its default arguments.
    """

    _lib_cache = {}
    
    __slots__ = [ "tree", "depth", "history", "_update_history", "lib" ]
    def __init__(self, depth, history = None, alphabet_size = 2):
        if alphabet_size not in self._lib_cache:
            ffi = FFI()
            ffi.cdef(re.sub("ALPHABET_SIZE", str(alphabet_size), ctwnode_h))
            self._lib_cache[alphabet_size] = ffi.verify(re.sub("ALPHABET_SIZE", str(alphabet_size), ctwnode_c))
        self.lib = self._lib_cache[alphabet_size]
        self.depth = depth
        self.history = history if history is not None else []
        self._update_history = history is None
        self.tree = self.lib.ctwnode_new()
        
    def __del__(self):
        self.lib.ctwnode_free(self.tree)

    def _mkcontext(self, x):
        padding = self.depth - len(x)
        return bytes([0] * padding + x[-self.depth:])
        
    def update(self, symbol, weight = 1.0):
        context = self._mkcontext(self.history)
        if self._update_history: self.history.append(symbol)
        return self.lib.ctwnode_update(self.tree, bytes([symbol]), weight, context, self.depth)

    def log_predict(self, symbol):
        context = self._mkcontext(self.history)
        return self.lib.ctwnode_log_predict(self.tree, bytes([symbol]), context, self.depth)

    @property
    def size(self):
        return self.lib.ctwnode_size(self.tree)

    def copy(self):
        cls = self.__class__
        r = cls.__new__(cls)
        r.lib = self.lib
        r.depth = self.depth
        r.history = self.history
        r._update_history = self._update_history
        r.tree = self.lib.ctwnode_copy(self.tree)
        return r
        
