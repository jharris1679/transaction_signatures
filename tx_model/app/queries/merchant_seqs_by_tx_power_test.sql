with transactions as
(select
  user_reference,
  a.transaction_id,
  auth_ts
from `tensile-oarlock-191715.postgres_reporting.transactions` a
join `tensile-oarlock-191715.user_personas.clustered_users` b on a.user_reference = b.index
join `koho-staging.merchant_vectors.int_indexed_common_raw_merchants` c
  on a.merchant_name = c.normalized_merchant_name
  and a.mcc = c.mcc
where description = 'Visa settle'
and label in (1, 11)
and merchant_name is not null
and sys_category is not null
and length(merchant_name) > 1
and amount*-1 > 0
and amount*-1 < 3000
and format_date('%Y-%m', date(auth_ts)) = '2020-02')


select
  a.auth_ts,
  a.user_reference,
  array_agg(lower(merchant_name) order by b.auth_ts desc limit {{seq_len}}) merchant_name,
  array_agg(cast(day_of_week as int64) order by b.auth_ts desc limit {{seq_len}}) day_of_week,
  array_agg(cast(eighth_of_day as int64) order by b.auth_ts desc limit {{seq_len}}) eighth_of_day,
  array_agg(cast(amount as float64) order by b.auth_ts desc limit {{seq_len}}) amount
from transactions a
join
  (select
    user_reference,
    transaction_id,
    merchant_name,
    auth_ts,
    extract(dayofweek from auth_ts) day_of_week,
    ceiling((extract(hour from auth_ts) + 1) / 3) as eighth_of_day,
    a.amount*-1 as amount,
    sys_category
  from `tensile-oarlock-191715.postgres_reporting.transactions` a
  join `koho-staging.merchant_vectors.int_indexed_common_raw_merchants` b
    on a.merchant_name = b.normalized_merchant_name
    and a.mcc = b.mcc
  where description = 'Visa settle'
  and merchant_name is not null
  and sys_category is not null
  and length(merchant_name) > 1
  and amount*-1 > 0
  and amount*-1 < 3000
  and date(auth_ts) = '2020-02-28') b
    on a.user_reference = b.user_reference
    and a.auth_ts > b.auth_ts
group by a.auth_ts, a.user_reference
having array_length(merchant_name) > 1
