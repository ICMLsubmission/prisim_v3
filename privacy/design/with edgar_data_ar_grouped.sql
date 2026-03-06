with edgar_data_ar_grouped (
    select
        uapathorig,
        uapathdest,
        psgrtype,
        cabin,
        itinuapathdepdt,
        bookdtdfd,
        avg(Error) as Error,
        avg(edgarrff) as edgarrff
    from
        (
            select
                uapathorig,
                uapathdest,
                itinfltnbr1,
                itinuapathdepdt,
                psgrtype,
                bkgs,
                cabin,
                bookdtdfd,
                edgarrff,
                nz_bkg_ind,
                bkg_date,
                multiplier,
                (bkgs - edgarrff) as Error
            from
                edgar_data_ar
            where
                itinuapathdepdt >= cast('{0}' as date)
        )
    group by
        uapathorig,
        uapathdest,
        psgrtype,
        cabin,
        itinuapathdepdt,
        bookdtdfd
),
edgar_data_snp_i as (
    select
        *,
        AVG(bkgs_prev_year) OVER(
            PARTITION BY uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            bookdtdfd,
            snpshtdtdfd
        ) as avg_prev_bkgs,
        AVG(rff_prev_year) OVER(
            PARTITION BY uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            bookdtdfd,
            snpshtdtdfd
        ) as avg_prev_rff
    from
        (
            select
                *,
                lag(bkg_avg, 1) over(
                    partition by uapathorig,
                    uapathdest,
                    bookdtdfd,
                    month(to_date(itinuapathdepdt))
                    order by
                        year(to_date(itinuapathdepdt))
                ) bkgs_prev_year,
                lag(rff_avg, 1) over(
                    partition by uapathorig,
                    uapathdest,
                    bookdtdfd,
                    month(to_date(itinuapathdepdt))
                    order by
                        year(to_date(itinuapathdepdt))
                ) rff_prev_year
            from
(
                    select
                        *,
                        case
                            when snpshtdtdfd = 7 then '1' --    when snpshtdtdfd = 14 then '2'
                            when snpshtdtdfd = 28 then '4'
                            when snpshtdtdfd = 35 then '5'
                            when snpshtdtdfd = 42 then '6'
                            when snpshtdtdfd = 70 then '7'
                            when snpshtdtdfd = 140 then '9'
                            else '-1'
                        end as m_col,
                        AVG(bkgs) over(
                            partition by uapathorig,
                            uapathdest,
                            bookdtdfd,
                            year(to_date(itinuapathdepdt)),
                            month(to_date(itinuapathdepdt))
                        ) as bkg_avg,
                        AVG(edgarrff) over(
                            partition by uapathorig,
                            uapathdest,
                            bookdtdfd,
                            year(to_date(itinuapathdepdt)),
                            month(to_date(itinuapathdepdt))
                        ) as rff_avg
                    from
                        edgar_data_snp
                    where
                        snpshtdtdfd in (7, 28, 35, 42, 70, 140)
                )
        )
    where
        itinuapathdepdt >= cast('{0}' as date)
),
edgar_data_ar_grouped_upd as (
    select
        *,
        case
            when bookdtdfd = 13 then '1' --    when bookdtdfd = 20 then '2'
            when bookdtdfd = 34 then '4'
            when bookdtdfd = 41 then '5'
            when bookdtdfd = 48 then '6'
            when bookdtdfd = 76 then '7'
            when bookdtdfd = 146 then '9'
            else '-1'
        end as m_col,
        case
            when bookdtdfd in (13, 34, 41, 48, 76, 146) then lag(Error, -1) over(
                partition by uapathorig,
                uapathdest,
                psgrtype,
                cabin,
                itinuapathdepdt
                order by
                    bookdtdfd
            )
            else null
        end as error_hori_lag1,
        case
            when bookdtdfd in (13, 34, 41, 48, 76, 146) then lag(Error, -2) over(
                partition by uapathorig,
                uapathdest,
                psgrtype,
                cabin,
                itinuapathdepdt
                order by
                    bookdtdfd
            )
            else null
        end as error_hori_lag2,
        case
            when bookdtdfd in (13, 34, 41, 48, 76, 146) then lag(Error, -3) over(
                partition by uapathorig,
                uapathdest,
                psgrtype,
                cabin,
                itinuapathdepdt
                order by
                    bookdtdfd
            )
            else null
        end as error_hori_lag3,
        case
            when bookdtdfd in (13, 34, 41, 48, 76, 146) then lag(Error, -4) over(
                partition by uapathorig,
                uapathdest,
                psgrtype,
                cabin,
                itinuapathdepdt
                order by
                    bookdtdfd
            )
            else null
        end as error_hori_lag4,
        case
            when bookdtdfd in (13, 34, 41, 48, 76, 146) then lag(Error, -5) over(
                partition by uapathorig,
                uapathdest,
                psgrtype,
                cabin,
                itinuapathdepdt
                order by
                    bookdtdfd
            )
            else null
        end as error_hori_lag5,
        case
            when bookdtdfd in (13, 34, 41, 48, 76, 146) then lag(Error, -6) over(
                partition by uapathorig,
                uapathdest,
                psgrtype,
                cabin,
                itinuapathdepdt
                order by
                    bookdtdfd
            )
            else null
        end as error_hori_lag6,
        case
            when bookdtdfd in (13, 34, 41, 48, 76, 146) then lag(Error, -7) over(
                partition by uapathorig,
                uapathdest,
                psgrtype,
                cabin,
                itinuapathdepdt
                order by
                    bookdtdfd
            )
            else null
        end as error_hori_lag7
    from
        edgar_data_ar_grouped
    where
        itinuapathdepdt >= cast('{0}' as date)
    order by
        uapathorig,
        uapathdest,
        psgrtype,
        cabin,
        itinuapathdepdt,
        bookdtdfd
),
edgar_data_snp_grouped (
    select
        uapathorig,
        uapathdest,
        psgrtype,
        cabin,
        itinuapathdepdt,
        snpshtdtdfd,
        bookdtdfd,
        avg(Error) as Error,
        avg(edgarrff) as edgarrff
    from
        (
            select
                uapathorig,
                uapathdest,
                itinfltnbr1,
                itinuapathdepdt,
                snpshtdtdfd,
                psgrtype,
                bkgs,
                cabin,
                bookdtdfd,
                edgarrff,
                nz_bkg_ind,
                bkg_date,
                multiplier,
                (bkgs - edgarrff) as Error
            from
                edgar_data_snp
            where
                itinuapathdepdt >= cast('{0}' as date)
        )
    group by
        uapathorig,
        uapathdest,
        psgrtype,
        cabin,
        itinuapathdepdt,
        bookdtdfd,
        snpshtdtdfd
),
edgar_data_vert as (
    select
        distinct uapathorig,
        uapathdest,
        psgrtype,
        cabin,
        itinuapathdepdt,
        case
            when snpshtdtdfd = 42 then 7 --    when snpshtdtdfd = 70 then 14
            when snpshtdtdfd = 70 then 28 --    when snpshtdtdfd = 70 then 35
            when snpshtdtdfd = 98 then 42
            when snpshtdtdfd = 140 then 70
            when snpshtdtdfd = 224 then 140
        end as join_condition,
        avg(Error) over(
            partition by uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
        ) as mean_error,
        avg(edgarrff) over(
            partition by uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
        ) as mean_edgarrff
    from
        edgar_data_snp_grouped
    where
        snpshtdtdfd in (42, 70, 98, 140, 224)
        and itinuapathdepdt >= cast('{0}' as date)
        and case
            when snpshtdtdfd = 42 then bookdtdfd >= 14
            and bookdtdfd <= 41 --    when snpshtdtdfd = 70 then bookdtdfd >= 21
            --    and bookdtdfd <= 48
            when snpshtdtdfd = 70 then bookdtdfd >= 35
            and bookdtdfd <= 62 --    when snpshtdtdfd = 70 then bookdtdfd >= 42
            --    and bookdtdfd <= 69
            when snpshtdtdfd = 98 then bookdtdfd >= 49
            and bookdtdfd <= 76
            when snpshtdtdfd = 140 then bookdtdfd >= 77
            and bookdtdfd <= 104
            when snpshtdtdfd = 224 then bookdtdfd >= 147
            and bookdtdfd <= 174
        end
    union
    all
    select
        distinct uapathorig,
        uapathdest,
        psgrtype,
        cabin,
        itinuapathdepdt,
        case
            when snpshtdtdfd = 70 then 35
        end as join_condition,
        avg(Error) over(
            partition by uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
        ) as mean_error,
        avg(edgarrff) over(
            partition by uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
        ) as mean_edgarrff
    from
        edgar_data_snp_grouped
    where
        snpshtdtdfd in (70)
        and itinuapathdepdt >= cast('{0}' as date)
        and case
            when snpshtdtdfd = 70 then bookdtdfd >= 42
            and bookdtdfd <= 69
        end
),
edgar_prev_bkgs_tbl as (
    select
        distinct uapathorig,
        uapathdest,
        psgrtype,
        cabin,
        itinuapathdepdt,
        bookdtdfd,
        snpshtdtdfd,
        Lag(avg_prev_bkgs, -1) OVER (
            PARTITION BY uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
            order by
                bookdtdfd
        ) as avg_prev_bkgs_lag1,
        Lag(avg_prev_bkgs, -2) OVER (
            PARTITION BY uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
            order by
                bookdtdfd
        ) as avg_prev_bkgs_lag2,
        Lag(avg_prev_bkgs, -3) OVER (
            PARTITION BY uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
            order by
                bookdtdfd
        ) as avg_prev_bkgs_lag3,
        Lag(avg_prev_bkgs, -4) OVER (
            PARTITION BY uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
            order by
                bookdtdfd
        ) as avg_prev_bkgs_lag4,
        Lag(avg_prev_bkgs, -5) OVER (
            PARTITION BY uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
            order by
                bookdtdfd
        ) as avg_prev_bkgs_lag5,
        Lag(avg_prev_bkgs, -6) OVER (
            PARTITION BY uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
            order by
                bookdtdfd
        ) as avg_prev_bkgs_lag6,
        Lag(avg_prev_bkgs, -7) OVER (
            PARTITION BY uapathorig,
            uapathdest,
            psgrtype,
            cabin,
            itinuapathdepdt,
            snpshtdtdfd
            order by
                bookdtdfd
        ) as avg_prev_bkgs_lag7
    from
        (
            select
                distinct uapathorig,
                uapathdest,
                psgrtype,
                cabin,
                itinuapathdepdt,
                avg_prev_bkgs,
                avg_prev_rff,
                bookdtdfd,
                snpshtdtdfd
            from
                edgar_data_snp_i
            where
                itinuapathdepdt >= cast('{0}' as date)
        )
)
select
    *,
    allraf2 / user_adj as RUF
from
    (
        select
            a.uapathorig,
            a.uapathdest,
            a.itinuapathdepdt,
            a.cabin,
            a.psgrtype,
            a.snpshtdtdfd,
            a.bookdtdfd,
            a.multiplier,
            a.MOY as MOY,
            a.DOW as DOW,
            a.bkg_date,
            a.edgarrff,
            a.POC,
            (a.bkgs - a.edgarrff) as Error,
            a.bkgs_prev_year,
            a.avg_prev_rff,
            a.itinfltcnt,
            avg_prev_bkgs_lag1,
            avg_prev_bkgs_lag2,
            avg_prev_bkgs_lag3,
            avg_prev_bkgs_lag4,
            avg_prev_bkgs_lag5,
            avg_prev_bkgs_lag6,
            avg_prev_bkgs_lag7,
            a.itinfltnbr1,
            a.nz_bkg_ind,
            a.bkgs,
            date_part('dow', a.bkg_date) as bkg_date_dow,
            a.entity,
            a.edgarraf2,
            a.bookprd,
            a.allraf2,
            a.edgarraf2 / a.edgarrff as user_adj,
            error_hori_lag1,
            error_hori_lag2,
            error_hori_lag3,
            error_hori_lag4,
            error_hori_lag5,
            error_hori_lag6,
            error_hori_lag7,
            c.mean_error,
            c.mean_edgarrff
        from
            edgar_data_snp_i as a
            left join edgar_data_ar_grouped_upd as b on a.uapathorig = b.uapathorig
            and a.uapathdest = b.uapathdest
            and a.itinuapathdepdt = b.itinuapathdepdt
            and a.cabin = b.cabin
            and a.psgrtype = b.psgrtype
            and a.m_col = b.m_col
            left join edgar_data_vert as c on a.uapathorig = c.uapathorig
            and a.uapathdest = c.uapathdest
            and a.itinuapathdepdt = c.itinuapathdepdt
            and a.cabin = c.cabin
            and a.psgrtype = c.psgrtype
            and a.snpshtdtdfd = c.join_condition
            left join edgar_prev_bkgs_tbl d on a.uapathorig = d.uapathorig
            and a.uapathdest = d.uapathdest
            and a.itinuapathdepdt = d.itinuapathdepdt
            and a.cabin = d.cabin
            and a.psgrtype = d.psgrtype
            and a.bookdtdfd = d.bookdtdfd
            and a.snpshtdtdfd = d.snpshtdtdfd
        where
            a.itinuapathdepdt >= cast('{0}' as date)
    )