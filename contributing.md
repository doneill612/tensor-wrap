# Contributing to tensor-wrap
Thanks for contributing! Your interest in the project is greatly appreciated.
## Bugs
- #### Find one?
  - **If you find a bug, create an [Issue](https://github.com/doneill612/tensor-wrap/issues)**
    that details the nature of the bug and how you encountered it. If possible,
    include a code-snippet or create a runnable test case which can be used to study
    the issue.
- #### Fix one?
  - **Open a [pull request](https://github.com/doneill612/tensor-wrap/pulls)**. Make
    sure the pull request has a clear and concise description of both the problem
    and your proposed solution.

## Features
If you design a new feature and are happy with the result,
feel free to open a [pull request](https://github.com/doneill612/tensor-wrap/pulls).
Make sure the pull request has a clear and concise description of the feature
you are adding, and make sure that it includes appropriate unit tests and results.

### Code style
This repository doesn't follow any particular Python coding standard. However,
try to abide by some of these general preferences:
- Use tabular indentation
- Keep lines between 65-80 characters.
- When breaking lines in method signatures, align arguments.
```python
  """ Correct """
  def _build_optimizer_ops(self, logits, v_logits,
                           labels, v_labels, learning_rate) -> None:
    pass

  """ Incorrect """
  def _build_optimizer_ops(self, logits, v_logits,
                                    labels, v_labels, learning_rate) -> None:
      pass
```
- Private methods **don't** require documentation, public methods **DO**.
```python
  def public_method(arg1: str, arg2: bool) -> bool:
    """
    Documenting this public method. Documentation between double
    quotes (""" """). Document arguments and return types as follows.

    Args:
      arg1 : arg1 description
      arg2 : arg2 description
    Returns:
      bool : description of return variable
    """
    ...
    ...

  def _private_method(arg1: int, arg2: int) -> None:
    ...
    ...
```
- Using Python 3.6, so give type hints whenever possible.

## Questions
Don't make an Issue for a specific question about the project. For now, simply
contact David O'Neill.
- david.oneill@businessdecision.com
- doneill612@gmail.com
