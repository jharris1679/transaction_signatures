with transactions as
(select
  user_reference,
  a.transaction_id,
  a.mcc,
  auth_ts
from `tensile-oarlock-191715.postgres_reporting.transactions` a
join `koho-staging.merchant_vectors.int_indexed_common_raw_merchants` b
  on a.merchant_name = b.normalized_merchant_name
  and a.mcc = b.mcc
where description = 'Visa settle'
and merchant_name is not null
and length(merchant_name) > 1
and format_date('%Y-%m', date(auth_ts)) <= '2019-12')


select
  a.auth_ts,
  array_agg(cast(b.mcc as string) order by b.auth_ts desc limit {{seq_len}}) merchant_sequence,
  array_agg(b.auth_ts order by b.auth_ts desc limit {{seq_len}}) timestamps
from transactions a
join
  (select
    user_reference,
    transaction_id,
    merchant_name,
    b.mcc,
    auth_ts
  from `tensile-oarlock-191715.postgres_reporting.transactions` a
  join `koho-staging.merchant_vectors.int_indexed_common_raw_merchants` b
    on a.merchant_name = b.normalized_merchant_name
    and a.mcc = b.mcc
  where description = 'Visa settle'
  and merchant_name is not null
  and length(merchant_name) > 1
  and format_date('%Y-%m', date(auth_ts)) <= '2019-12') b
    on a.user_reference = b.user_reference
    and a.auth_ts > b.auth_ts
group by a.auth_ts
having array_length(timestamps) > 1
