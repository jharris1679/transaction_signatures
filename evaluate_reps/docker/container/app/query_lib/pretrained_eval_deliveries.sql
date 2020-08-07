with target as
(select
  a.created_at,
  b.user_reference,
  case
    when regexp_contains(merchant_name, r'(?i)doordash') then 'doordash'
    when regexp_contains(merchant_name, r'(?i)skipthedishes') then 'skipthedishes'
    when regexp_contains(merchant_name, r'(?i)uber eats') then 'uber eats'
  end as merchant_type,
  farm_fingerprint(concat(cast(created_at as string), cast(account_number as string))) shuffle
from `tensile-oarlock-191715.kohoapi.authorizations_stream` a
join `tensile-oarlock-191715.postgres_reporting.user_identifier_lookup` b on cast(a.account_number as string) = b.prn
where event_type = 'AuthorizationApproved'
and regexp_contains(merchant_name, r'(?i)doordash|skipthedishes|uber eats')
order by shuffle
limit 10000),

counter_examples as
(select
  a.created_at,
  b.user_reference,
  'not delivery' as merchant_type,
  farm_fingerprint(concat(cast(created_at as string), cast(account_number as string))) shuffle
from `tensile-oarlock-191715.kohoapi.authorizations_stream` a
join `tensile-oarlock-191715.postgres_reporting.user_identifier_lookup` b on cast(a.account_number as string) = b.prn
where event_type = 'AuthorizationApproved'
and not regexp_contains(merchant_name, r'(?i)doordash|skipthedishes|uber eats')
order by shuffle
limit 10000),

embedded_auths as
(select
  a.created_at auth_created_at,
  a.account_number,
  c.* except(index)
from `tensile-oarlock-191715.kohoapi.authorizations_stream` a
join `tensile-oarlock-191715.postgres_reporting.transactions` b on a.authorization_id = b.auth_source_id
join `merchant-embeddings.project_report.pretrained_merchant_embeddings` c on lower(b.merchant_name) = c.merchant_name
where event_type = 'AuthorizationApproved')

select
  a.* except(shuffle),
  c.* except(account_number, merchant_name, auth_created_at)
from target a
join `tensile-oarlock-191715.postgres_reporting.user_identifier_lookup` b using(user_reference)
join embedded_auths c on b.prn = cast(c.account_number as string) and date(c.auth_created_at) between date_sub(date(a.created_at), interval 60 day) and date(a.created_at)

union all

select
  a.* except(shuffle),
  c.* except(account_number, merchant_name, auth_created_at)
from counter_examples a
join `tensile-oarlock-191715.postgres_reporting.user_identifier_lookup` b using(user_reference)
join embedded_auths c on b.prn = cast(c.account_number as string) and date(c.auth_created_at) between date_sub(date(a.created_at), interval 60 day) and date(a.created_at)
