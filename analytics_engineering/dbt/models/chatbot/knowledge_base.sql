{{
    config(
        materialized='table'
    )
}}

select * from {{ref("transform_blurt")}}
union
select * from {{ref("transform_chipur")}}
union
select * from {{ref("transform_nat")}}