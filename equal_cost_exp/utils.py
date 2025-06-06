import inspect
# import collections
import collections.abc
import pandas as pd

# See https://www.python-course.eu/python3_memoization.php
class Memoize:

  def __init__(self, fn):

    self.fn = fn
    self.memo = {}

  def __call__(self, *args, **kwargs):

    sig = inspect.signature(self.fn)
    ba = sig.bind(*args)
    for param in sig.parameters.values():
      # to support default args: https://docs.python.org/3.3/library/inspect.html
      if param.name not in ba.arguments:
        ba.arguments[param.name] = param.default
      # For some reason, when we pass in something like tmp(a,b=1) to the function
      # def tmp(a,b=0), the args will remain (a,0) but not (a,1). Therefore, we
      # should update `ba` according to the k,v pairs in kwargs
      if param.name in kwargs.keys():
        ba.arguments[param.name] = kwargs[param.name]
    args = ba.args

    # # convert DataFrames (not hashable because mutable) to numpy array (hashable)
    # args = [
    #   elem.copy().to_numpy() if isinstance(elem, pd.DataFrame) else elem
    #   for elem in args
    # ]

    # convert lists and numpy array into tuples so that they can be used as keys
    hashable_args = tuple([
      # collections.Hashable = collections.abc.Hashable
      arg if isinstance(arg, collections.abc.Hashable) else str(arg)
      for arg in args
    ])

    if hashable_args not in self.memo:
      self.memo[hashable_args] = self.fn(*args)
    return self.memo[hashable_args]

def convertToOneHotWithPrespecifiedCategories(df, node, lower_bound, upper_bound):
  return pd.get_dummies(
    pd.Categorical(
      df[node],
      categories=list(range(
        int(lower_bound),
        int(upper_bound) + 1
      ))
    ),
    prefix=node
  )