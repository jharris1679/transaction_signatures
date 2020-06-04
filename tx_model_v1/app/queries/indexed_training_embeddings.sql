select
  index,
  lower(b.merchant_name) merchant_name,
  b.* except(merchant_name)
from `koho-staging.merchant_vectors.int_indexed_common_raw_merchants`  a
join `koho-staging.merchant_vectors.gensim_embedding`  b on a.normalized_merchant_name = b.merchant_name
order by index
