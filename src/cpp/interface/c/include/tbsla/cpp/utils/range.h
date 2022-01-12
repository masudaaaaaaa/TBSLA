#ifndef TBSLA_CINTERFACE_CPP_range
#define TBSLA_CINTERFACE_CPP_range

#ifdef __cplusplus
extern "C" {
#endif

static int lnv(int s, int l, int g)
{
  int n = s / g;
  int mod = s % g;
  if (l < mod)
    n++;
  return n;
}

static int pflv(int s, int l, int g)
{
  int mod = s % g;
  int n = lnv(s, l, g) * l;
  if (l >= mod)
    n += mod;
  return n;
}

#ifdef __cplusplus
}
#endif

#endif /* TBSLA_CINTERFACE_CPP_Vector */
