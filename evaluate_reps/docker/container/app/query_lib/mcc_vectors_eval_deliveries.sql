with target as
(select
  a.created_at,
  b.user_reference,
  'Foodora' as merchant_type,
  farm_fingerprint(concat(cast(created_at as string), cast(account_number as string))) shuffle
from `tensile-oarlock-191715.kohoapi.authorizations_stream` a
join `tensile-oarlock-191715.postgres_reporting.user_identifier_lookup` b on cast(a.account_number as string) = b.prn
where event_type = 'AuthorizationApproved'
and regexp_contains(merchant_name, r'(?i)foodora')
order by shuffle),

counter_examples as
(select
  a.created_at,
  b.user_reference,
  'Not Foodora' as merchant_type,
  farm_fingerprint(concat(cast(created_at as string), cast(account_number as string))) shuffle
from `tensile-oarlock-191715.kohoapi.authorizations_stream` a
join `tensile-oarlock-191715.postgres_reporting.user_identifier_lookup` b on cast(a.account_number as string) = b.prn
where event_type = 'AuthorizationApproved'
and not regexp_contains(merchant_name, r'(?i)foodora')
order by shuffle
limit 8500),

mcc_vec_auths as
(select
  a.created_at auth_created_at,
  a.account_number,
  b.*
from `tensile-oarlock-191715.kohoapi.authorizations_stream` a
join `tensile-oarlock-191715.staging_tables.mcc_vectors` b on cast(a.mcc as string) = b.mcc
where event_type = 'AuthorizationApproved')

select
  a.* except(shuffle),
  c.* except(account_number, mcc, description, auth_created_at)
from target a
join `tensile-oarlock-191715.postgres_reporting.user_identifier_lookup` b using(user_reference)
join mcc_vec_auths c on b.prn = cast(c.account_number as string) and date(c.auth_created_at) between date_sub(date(a.created_at), interval 60 day) and date(a.created_at)

union all

select
  a.* except(shuffle),
  c.* except(account_number, mcc, description, auth_created_at)
from counter_examples a
join `tensile-oarlock-191715.postgres_reporting.user_identifier_lookup` b using(user_reference)
join mcc_vec_auths c on b.prn = cast(c.account_number as string) and date(c.auth_created_at) between date_sub(date(a.created_at), interval 60 day) and date(a.created_at)
