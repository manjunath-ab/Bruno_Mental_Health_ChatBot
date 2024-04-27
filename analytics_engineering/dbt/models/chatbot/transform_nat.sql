{{
    config(
        materialized='view'
    )
}}

with nat as
(

select 
 MENTAL_ILLNESS_TITLE,
 case COPING_MECHANISM
 when 'N/A' then 'No coping mechanism provided'
 when null then 'No coping mechanism provided'
 else COPING_MECHANISM
 end as COPING_MECHANISM,
 case SUPPORT_SYSTEM
 when 'N/A' then 'No support system provided'
 when null then 'No support system provided'
 else SUPPORT_SYSTEM
 end as SUPPORT_SYSTEM,
 case TRIGGERS
 when 'N/A' then 'No triggers provided'
 when null then 'No triggers provided'
 else TRIGGERS
 end as TRIGGERS,
 case REFLECTIONS
 when 'N/A' then 'No reflection provided'
 when null then 'No reflection provided'
 else REFLECTIONS
 end as REFLECTIONS,
 case SELF_CARE_PRACTICES
 when 'N/A' then 'No self care practices  provided'
 when null then 'No self care practices  provided'
 else SELF_CARE_PRACTICES
 end as SELF_CARE_PRACTICES

 



 from {{source('chatbot','nat')}} where MENTAL_ILLNESS_TITLE != 'N/A' and MENTAL_ILLNESS_TITLE is not null

)
select * from nat
