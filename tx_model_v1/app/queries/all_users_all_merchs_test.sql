with users as
(select distinct
  user_reference
from `tensile-oarlock-191715.postgres_reporting.transactions` a
join `tensile-oarlock-191715.user_personas.clustered_users` b on a.user_reference = b.index
where description = 'Visa settle'
and merchant_name is not null
and sys_category is not null
and length(merchant_name) > 1
and amount*-1 > 0
and amount*-1 < 3000
and format_date('%Y-%m', date(auth_ts)) = '2020-04')


select
  a.user_reference,
  array_agg(lower(merchant_name) order by auth_ts desc) merchant_name,
  array_agg(cast(day_of_week as int64) order by auth_ts desc) day_of_week,
  array_agg(cast(eighth_of_day as int64) order by auth_ts desc) eighth_of_day,
  array_agg(cast(amount as float64) order by auth_ts desc) amount,
  array_agg(cast(mcc as int64) order by auth_ts desc) mcc
from users a
join
  (select
    user_reference,
    transaction_id,
    merchant_name,
    auth_ts,
    extract(dayofweek from auth_ts) day_of_week,
    ceiling((extract(hour from auth_ts) + 1) / 3) as eighth_of_day,
    a.amount*-1 as amount,
    sys_category,
    a.mcc
  from `tensile-oarlock-191715.postgres_reporting.transactions` a
  where description = 'Visa settle'
  and merchant_name is not null
  and sys_category is not null
  and length(merchant_name) > 1
  and amount*-1 > 0
  and amount*-1 < 3000
  and format_date('%Y-%m', date(auth_ts)) = '2020-04') b using(user_reference)
group by a.user_reference
having array_length(merchant_name) > 1
