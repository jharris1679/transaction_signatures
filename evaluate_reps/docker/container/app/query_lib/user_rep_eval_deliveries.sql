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
limit 10000)

select
  a.* except(shuffle),
  b.* except(int64_field_0, string_field_1)
from target a
join `merchant-embeddings.project_report.referral_disturbed_user_embedding` b on a.user_reference = b.string_field_1

union all

select
  a.* except(shuffle),
  b.* except(int64_field_0, string_field_1)
from counter_examples a
join `merchant-embeddings.project_report.referral_disturbed_user_embedding` b on a.user_reference = b.string_field_1
