#pragma once

#include "sys/time.h"
#include <ctime>

double now()
{
  struct timeval t;
  double f_t;
  gettimeofday(&t, NULL);
  f_t = t.tv_usec;
  f_t = f_t / ((float)1E6);
  f_t += t.tv_sec;
  return (f_t);
}