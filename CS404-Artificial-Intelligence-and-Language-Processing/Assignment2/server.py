import os 
from typing import Any, Dict, List, Optional, Literal
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv
from fastmcp import FastMCP

##load config .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

##create mcp instance
mcp=FastMCP("Steam Games MCP Server")

def connect_mysql():
    '''
    Create MySQL connection for each tool call.
    '''
    ##get mysql configuration
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DB", "steam_games")

    ##connect Mysql
    steam_game_db = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=db,
        charset="utf8mb4",
        use_unicode=True,
        cursorclass=DictCursor,
        autocommit=True,
    )

    return steam_game_db


def fetch_all(sql:str, params:tuple=())->List[Dict[str,Any]]:
    '''
    sql:the sql sequences executed
    params:the placeholder of sql sequences
    return: a list include many dicts
    '''

    ##get a database connection, and if complete mysql, automatically close link, submit and rollback
    with connect_mysql() as steam_game:
        ##create cursor to execute 
        with steam_game.cursor() as cur:
            ##using utf8 to encode
            cur.execute("SET NAMES utf8mb4;")
            ##execute sql statement
            cur.execute(sql,params)
            ##get all result
            return list(cur.fetchall())

def fetch_one(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    '''
    get the first result
    '''
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


##Tool1:Filter games with common attributions
@mcp.tool
def search_game(
    game_name_kw:Optional[str] = None,
    publisher: Optional[str] = None,
    developer: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    start_date: Optional[str] = None, 
    platform: Optional[Literal["windows", "mac", "linux"]] = None,
    min_positive: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
)-> Dict[str, Any]:
    '''
    game_name_kw: Game name keyword
    publisher: The name of publisher
    developer: The name of developer
    min_price: The minimal/discounted price
    max_price: The maximal/original price
    start_date: The published date
    platform: if the game supports window/mac/linux
    min_positive: positive vote, because people want to play good game whose the number of the votes is greater than certain value
    limit: print results
    offset: how much the result return every slide

    return :result
    '''

    ##limit the number of results
    limit=max(1, min(int(limit),200))
    offset = max(0, int(offset))

    ##sql statement list
    where=[]
    params:List[Any]=[]

    ##game name keyword query
    if game_name_kw:
        where.append("game_name LIKE %s")
        params.append(f"%{game_name_kw}%")

    ##publisher name keyword query
    if publisher:
        where.append("publishers LIKE %s")
        params.append(f"%{publisher}%")


    ##developer name keyword query
    if developer:
        where.append("developers LIKE %s")
        params.append(f"%{developer}%")

    ##discounted_price
    if min_price is not None:
        where.append("price >= %s")
        params.append(float(min_price))

    ##original_price
    if max_price is not None:
        where.append("price <= %s")
        params.append(float(max_price))

    ##The release date is greater then input start date
    if start_date:
        where.append("release_date >= %s")
        params.append(start_date)

    ##platform
    if platform:
        col = {
            "windows": "support_window",
            "mac": "support_mac",
            "linux": "support_linux",
        }[platform]
        where.append(f"{col} = 1")

    ##positive votes number
    if min_positive is not None:
        where.append("positive >= %s")
        params.append(int(min_positive))

    ##Use AND to compose where conditions
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    ##total count
    count_sql = f"SELECT COUNT(*) AS total FROM games {where_sql}"
    total_row = fetch_one(count_sql, tuple(params)) or {"total": 0}

    ##query rows
    query_sql = f"""
        SELECT
            app_id, game_name, release_date,
            price, discount, dlc_count,
            required_age,
            support_window, support_mac, support_linux,
            positive, negative, recommendations,
            developers, publishers
        FROM games
        {where_sql}
        ORDER BY
            release_date DESC,
            positive DESC
        LIMIT %s OFFSET %s
    """
    rows = fetch_all(query_sql, tuple(params + [limit, offset]))

    return {
        "total": int(total_row["total"]),
        "limit": limit,
        "offset": offset,
        "rows": rows
    }
    

## search a game by appid
@mcp.tool
def get_game_by_app_id(app_id: int) -> Optional[Dict[str, Any]]:
    '''
    app_id: Steam AppID (primary key)
    return: one game record (dict) or None
    '''
    sql = "SELECT * FROM games WHERE app_id = %s"
    return fetch_one(sql, (int(app_id),))


##compute approval rate 
@mcp.tool
def top_games_by_approval(
    min_total_reviews: int = 50,
    platform: Optional[Literal["windows", "mac", "linux"]] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    '''
    min_total_reviews: minimal (positive + negative) for reliability
    platform: filter by support_window/support_mac/support_linux
    limit/offset: pagination
    return: total count and rows ordered by approval rate desc
    '''

    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    min_total_reviews = max(0, int(min_total_reviews))

    where = []
    params: List[Any] = []

    # require positive/negative not null and enough total reviews
    where.append("positive IS NOT NULL")
    where.append("negative IS NOT NULL")
    where.append("(positive + negative) >= %s")
    params.append(min_total_reviews)

    if platform:
        col = {
            "windows": "support_window",
            "mac": "support_mac",
            "linux": "support_linux",
        }[platform]
        where.append(f"{col} = 1")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    # total count
    count_sql = f"SELECT COUNT(*) AS total FROM games {where_sql}"
    total_row = fetch_one(count_sql, tuple(params)) or {"total": 0}

    # query rows with approval rate
    query_sql = f"""
        SELECT
            app_id, game_name, release_date, price, discount,
            positive, negative,
            (positive / NULLIF((positive + negative), 0)) AS approval_rate,
            support_window, support_mac, support_linux,
            publishers, developers
        FROM games
        {where_sql}
        ORDER BY approval_rate DESC, (positive + negative) DESC
        LIMIT %s OFFSET %s
    """
    rows = fetch_all(query_sql, tuple(params + [limit, offset]))

    return {
        "total": int(total_row["total"]),
        "limit": limit,
        "offset": offset,
        "rows": rows
    }


##In a year, the number of games released, the average price of all games released in this year and average discount 
@mcp.tool
def yearly_summary(
    start_year: int = 2005,
    end_year: int = 2026,
    platform: Optional[Literal["windows", "mac", "linux"]] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    '''
    start_year/end_year: filter by YEAR(release_date)
    platform: filter by support_window/support_mac/support_linux
    limit: max returned years (usually <= 200)
    return: rows grouped by year
    '''

    start_year = int(start_year)
    end_year = int(end_year)
    limit = max(1, min(int(limit), 500))

    where = []
    params: List[Any] = []

    where.append("release_date IS NOT NULL")
    where.append("YEAR(release_date) BETWEEN %s AND %s")
    params.extend([start_year, end_year])

    if platform:
        col = {
            "windows": "support_window",
            "mac": "support_mac",
            "linux": "support_linux",
        }[platform]
        where.append(f"{col} = 1")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        SELECT
            YEAR(release_date) AS year,
            COUNT(*) AS n_games,
            AVG(price) AS avg_price,
            AVG(discount) AS avg_discount,
            AVG(dlc_count) AS avg_dlc_count
        FROM games
        {where_sql}
        GROUP BY YEAR(release_date)
        ORDER BY year ASC
        LIMIT %s
    """
    rows = fetch_all(sql, tuple(params + [limit]))

    return {
        "start_year": start_year,
        "end_year": end_year,
        "platform": platform,
        "rows": rows
    }


##linux and nonlinx approval rate comparison
@mcp.tool
def compare_linux_vs_nonlinux(
    min_total_reviews: int = 50
) -> Dict[str, Any]:
    '''
    min_total_reviews: minimal (positive + negative) for reliability
    return: two groups (support_linux=1 and support_linux=0) comparison
    '''

    min_total_reviews = max(0, int(min_total_reviews))

    sql = """
        SELECT
            support_linux AS linux_supported,
            COUNT(*) AS n_games,
            SUM(positive) AS sum_positive,
            SUM(negative) AS sum_negative,
            AVG(positive / NULLIF((positive + negative), 0)) AS avg_approval_rate
        FROM games
        WHERE positive IS NOT NULL
          AND negative IS NOT NULL
          AND (positive + negative) >= %s
        GROUP BY support_linux
        ORDER BY linux_supported DESC
    """
    rows = fetch_all(sql, (min_total_reviews,))

    return {
        "min_total_reviews": min_total_reviews,
        "groups": rows
    }



##publisher publishes the number of games and average approval rate
@mcp.tool
def top_publishers(
    min_games: int = 10,
    min_total_reviews: int = 50,
    limit: int = 20
) -> Dict[str, Any]:
    '''
    min_games: publisher must have at least this many games
    min_total_reviews: each game must have (positive+negative) >= this threshold
    limit: return top N publishers by avg approval rate
    return: publisher ranking
    '''

    min_games = max(1, int(min_games))
    min_total_reviews = max(0, int(min_total_reviews))
    limit = max(1, min(int(limit), 200))

    sql = """
        SELECT
            publishers,
            COUNT(*) AS n_games,
            AVG(positive / NULLIF((positive + negative), 0)) AS avg_approval_rate,
            SUM(positive) AS sum_positive,
            SUM(negative) AS sum_negative
        FROM games
        WHERE publishers IS NOT NULL AND publishers <> ''
          AND positive IS NOT NULL AND negative IS NOT NULL
          AND (positive + negative) >= %s
        GROUP BY publishers
        HAVING COUNT(*) >= %s
        ORDER BY avg_approval_rate DESC, n_games DESC
        LIMIT %s
    """
    rows = fetch_all(sql, (min_total_reviews, min_games, limit))

    return {
        "min_games": min_games,
        "min_total_reviews": min_total_reviews,
        "limit": limit,
        "rows": rows
    }


if __name__ == "__main__":
    mcp.run()
