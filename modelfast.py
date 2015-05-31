from model import Model
from cffi import FFI

ffi = FFI()

ffi.cdef("""
typedef struct _ctwnode_t {
  double base_counts[2];
  double base_log_prob;
  double children_log_prob;
  double log_prob;
  struct _ctwnode_t *children[2];
  unsigned int _refcount;
} ctwnode_t;

double ctwnode_update(ctwnode_t *node, char symbol, double weight, char *context, int ctxtlen);
ctwnode_t *ctwnode_new();
ctwnode_t *ctwnode_copy(ctwnode_t *self);
void ctwnode_free(ctwnode_t *self);
void ctwnode_print(ctwnode_t *self);
int ctwnode_size(ctwnode_t *self);
""")

lib = ffi.verify("""
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct _ctwnode_t {
  double base_counts[2];
  double base_log_prob;
  double children_log_prob;
  double log_prob;
  struct _ctwnode_t *children[2];
  unsigned int _refcount;
} ctwnode_t;

double logsumexp(double a, double b)
{
  double shift = (a > b) ? a : b;
  return log(exp(a - shift) + exp(b - shift)) + shift;
}

ctwnode_t *ctwnode_new()
{
  ctwnode_t *self = malloc(sizeof(ctwnode_t));
  self->base_counts[0] = self->base_counts[1] = 0.5;
  self->base_log_prob = self->children_log_prob = self->log_prob = 0.0;
  self->children[0] = self->children[1] = NULL;
  self->_refcount = 1;
  return self;
}

void ctwnode_free(ctwnode_t *self)
{
  self->_refcount--;
  if (self->_refcount == 0) {
    if (self->children[0]) ctwnode_free(self->children[0]);
    if (self->children[1]) ctwnode_free(self->children[1]);
    free(self);
  }
}

ctwnode_t *ctwnode_copy(ctwnode_t *self)
{
  ctwnode_t *copy = malloc(sizeof(ctwnode_t));
  (*copy) = (*self);
  copy->_refcount = 1;
  if (self->children[0]) self->children[0]->_refcount++;
  if (self->children[1]) self->children[1]->_refcount++;
  return copy;
}
  
void ctwnode_base_update(ctwnode_t *self, char symbol, double weight)
{
  self->base_log_prob += log(self->base_counts[symbol] / (self->base_counts[0] + self->base_counts[1]));
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
    self->log_prob = log(0.5) + logsumexp(self->base_log_prob, self->children_log_prob);
  } else {
    self->log_prob = self->base_log_prob;
  }

  return self->log_prob - orig_log_prob;
}

double ctwnode_log_predict(ctwnode_t *self, char symbol, char *context, unsigned int ctxtlen)
{
  double base_log_prob = self->base_log_prob + log(self->base_counts[symbol] / (self->base_counts[0] + self->base_counts[1]));

  if (ctxtlen) {
    int cnext = context[ctxtlen-1];
    ctwnode_t *child =self->children[cnext];
    if (!child) child = self->children[cnext] = ctwnode_new();

    double children_log_prob = self->children_log_prob + ctwnode_log_predict(child, symbol, context, ctxtlen-1);

    return log(0.5) + logsumexp(base_log_prob, children_log_prob) - self->log_prob;
  } else {
    return base_log_prob - self->log_prob;
  }
}

void ctwnode_print(ctwnode_t *self)
{
  printf("%lf %lf %lf\\n", self->base_log_prob, 
    self->children[0] ? self->children[0]->base_log_prob : 0.0,
    self->children[1] ? self->children[1]->base_log_prob : 0.0);
}

int ctwnode_size(ctwnode_t *self)
{
  int size = 1;
  if (self->children[0]) size += ctwnode_size(self->children[0]);
  if (self->children[1]) size += ctwnode_size(self->children[1]);
  return size;
}
""")

class CTW_KT(Model):
    """Context Tree Weighting over KT models with a binary alphabet.

    A specialized memory and time efficient version of model.CTW with
    its default arguments.
    """
    
    __slots__ = [ "tree", "depth", "history", "_update_history" ]
    def __init__(self, depth, history = None):
        self.depth = depth
        self.history = history if history is not None else []
        self._update_history = history is None
        self.tree = lib.ctwnode_new()

    def __del__(self):
        lib.ctwnode_free(self.tree)

    def _mkcontext(self, x):
        padding = self.depth - len(x)
        return bytes([0] * padding + x[-self.depth:])
        
    def update(self, symbol, weight = 1.0):
        context = self._mkcontext(self.history)
        if self._update_history: self.history.append(symbol)
        return lib.ctwnode_update(self.tree, bytes([symbol]), weight, context, self.depth)

    def log_predict(self, symbol):
        return lib.ctwnode_log_predict(self.tree, bytes([symbol]), weight, bytes(self.history[-self.depth:]), self.depth)

    @property
    def size(self):
        return lib.ctwnode_size(self.tree)

    def copy(self):
        cls = self.__class__
        r = cls.__new__(cls)
        r.depth = self.depth
        r.history = self.history
        r._update_history = self._update_history
        r.tree = lib.ctwnode_copy(self.tree)
        return r
        
