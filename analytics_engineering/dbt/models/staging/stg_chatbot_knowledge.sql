{{
    config(
        materialized='table'
    )
}}

with source as 
(
  select *
  from {{ source('staging','chatbot_knowledge') }}
  
)

select * from stg_chatbot_knowledge
