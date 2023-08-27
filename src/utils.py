def formatNumbersWithZeros(number, nZeros):
  if type(number) == int:
    number = str(number)
  while len(number) < nZeros:
    number = "0" + number
  return number