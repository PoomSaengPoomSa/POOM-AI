import datetime
from db import get_db_cursor
from langsmith import traceable

def get_customer(customer_id: int):
    """
    Get customer asset portfolio details.
    """
    query = """
        SELECT c_id, name, total_assets, deposit, investment, pension, loan, net_worth, tendency, grade
        FROM customer
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        result = cursor.fetchone()
        return result

def get_trend_report():
    """
    Retrieve completed trend reports.
    """
    query = """
        (
            SELECT type, content, created_at
            FROM trend_llm_report
            WHERE type = 'gold' 
              AND DATE(created_at) = CURDATE()
            ORDER BY created_at DESC
            LIMIT 1
        )
        UNION ALL
        (
            SELECT r1.type, r1.content, r1.created_at
            FROM trend_llm_report r1
            WHERE r1.type != 'gold'
              AND YEAR(r1.created_at) = YEAR(CURDATE())
              AND MONTH(r1.created_at) = MONTH(CURDATE())
              AND r1.created_at = (
                  SELECT MAX(r2.created_at)
                  FROM trend_llm_report r2
                  WHERE r2.type = r1.type
                    AND YEAR(r2.created_at) = YEAR(CURDATE())
                    AND MONTH(r2.created_at) = MONTH(CURDATE())
              )
        )
    """
    with get_db_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
        return results

def get_customer_features(customer_id: int, months: int = 3):
    """
    Get customer features extracted from the database for the given period (months).
    """
    query = """
        SELECT category, contents, created_date
        FROM customer_information
        WHERE c_id = %s AND created_date >= DATE_SUB(NOW(), INTERVAL %s MONTH)
        ORDER BY created_date DESC
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id, months))
        results = cursor.fetchall()
        return results

def get_large_external_transactions(customer_id: int, threshold_amount: float = 10000000.0):
    """
    Retrieve external transactions where the customer transferred out a large amount of money.
    """
    query = """
        SELECT amount, opp_bank_name, briefs, ct_datetime, balance_after
        FROM customer_transaction
        WHERE c_id = %s AND opp_bank_name != '우리은행' AND ct_type = 'W' AND amount >= %s
          AND ct_datetime >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
        ORDER BY ct_datetime DESC
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id, threshold_amount))
        results = cursor.fetchall()
        return results

def save_asset_insight(customer_id: int, insight: str):
    """
    Save the LLM generated asset profile analysis result to customer's llm_insight column and update analysis_time.
    """
    query = """
        UPDATE customer
        SET llm_insight = %s, analysis_time = NOW()
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, (insight, customer_id))
        return rows_affected > 0

def save_churn_level(customer_id: int, grade: str, reason: str, explain_reason: str = ""):
    """
    Insert a new churn risk level assessment into churn_level table.
    """
    query = """
        INSERT INTO churn_level (c_id, grade, reason, explain_reason, created_date)
        VALUES (%s, %s, %s, %s, NOW())
    """
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, (customer_id, grade, reason, explain_reason))
        return rows_affected > 0

def get_recent_consultation_report(customer_id: int):
    """
    Get the latest consultation_report content for the customer.
    """
    query = """
        SELECT r.cr_id, r.key_contents, r.special_notes, r.follow_up_actions, r.summary, m.consult_date, m.u_id
        FROM consultation_report r
        JOIN consultation_memo m ON r.cm_id = m.cm_id
        WHERE m.c_id = %s
        ORDER BY m.consult_date DESC
        LIMIT 1
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        result = cursor.fetchone()
        if result:
            content_parts = []
            if result.get("key_contents"):
                content_parts.append(f"[핵심 상담 내용]\n{result['key_contents']}")
            if result.get("special_notes"):
                content_parts.append(f"[특이사항 및 추가 상담 계획]\n{result['special_notes']}")
            if result.get("follow_up_actions"):
                content_parts.append(f"[향후 조치 사항]\n{result['follow_up_actions']}")
            if result.get("summary"):
                content_parts.append(f"[요약]\n{result['summary']}")
            
            result["content"] = "\n\n".join(content_parts)
        return result

def get_main_products():
    """
    Retrieve active bank main products from the product table.
    """
    query = """
        SELECT pd_id, name, explanation, type, features, target_customer, expected_return, return_type
        FROM product
        WHERE is_main = 1
    """
    with get_db_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
        return results

def save_product_matching(product_id: int, customer_id: int, is_suitable: int, reason: str):
    """
    Upsert product matching suitability evaluation result.
    """
    delete_query = """
        DELETE FROM product_matching
        WHERE pd_id = %s AND c_id = %s
    """
    insert_query = """
        INSERT INTO product_matching (pd_id, c_id, is_suitable, reason, created_date)
        VALUES (%s, %s, %s, %s, NOW())
    """
    with get_db_cursor() as cursor:
        cursor.execute(delete_query, (product_id, customer_id))
        rows_affected = cursor.execute(insert_query, (product_id, customer_id, is_suitable, reason))
        return rows_affected > 0

def get_customer_relationship(customer_id: int):
    """
    Retrieve customer family relationships.
    """
    query = """
        SELECT relationship, information
        FROM customer_relationship
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        results = cursor.fetchall()
        return results

def get_customer_active_products(customer_id: int):
    """
    Retrieve products currently held by the customer.
    """
    query = """
        SELECT cp.pd_id, p.name as product_name, cp.opening_date, cp.expiration_date
        FROM customer_product cp
        JOIN product p ON cp.pd_id = p.pd_id
        WHERE cp.c_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        results = cursor.fetchall()
        return results

def get_customer_accounts(customer_id: int):
    """
    Retrieve customer's account types and balances.
    """
    query = """
        SELECT account_num, account_type, balance, opening_date
        FROM customer_account
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        results = cursor.fetchall()
        return results

def get_customer_transactions(customer_id: int, months: int = 3):
    """
    Retrieve all transactions for a specific customer in the last N months.
    """
    query = """
        SELECT amount, opp_bank_name, briefs, ct_datetime, balance_after, ct_type
        FROM customer_transaction
        WHERE c_id = %s AND ct_datetime >= DATE_SUB(NOW(), INTERVAL %s MONTH)
        ORDER BY ct_datetime DESC
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id, months))
        results = cursor.fetchall()
        return results

@traceable(name="fetch_batch_target_c_ids", run_type="tool")
def fetch_batch_target_c_ids() -> list:
    """
    DB 단일 스캔 쿼리를 통해 분석 후보 VVIP 고객 정보 및 스캔 조건 사유 추출
    1. 이탈 위험 수준이 '위험'인 고객
    2. 마지막 방문(상담) 이력이 30일 이상 경과(혹은 없음)한 고객
    3. 오늘 상담이 예정된 고객
    4. 최근 3개월 내에 타행 거액 거래 내역이 있는 고객
    5. 만기 예정 상품을 보유한 고객
    6. 고객 정보가 update된 이후에 AI 분석이 없었던 고객(customer.update_time과 analysis_time 열 확인)
    """
    query = """
        SELECT c.c_id, c.name, c.total_assets, c.net_worth, c.deposit, c.loan, r.reason
        FROM customer c
        JOIN (
            -- (1) 이탈 위험 수준이 '위험'인 고객
            SELECT c_id, '이탈 위험 수준 [위험]' AS reason
            FROM churn_level
            WHERE created_date = (
                SELECT MAX(created_date) FROM churn_level WHERE c_id = churn_level.c_id
            ) AND grade = '위험'
            
            UNION ALL
            
            -- (2) 마지막 방문(상담) 이력이 30일 이상 경과(혹은 없음)한 고객
            SELECT c.c_id, '마지막 상담 이력 30일 경과 또는 없음' AS reason
            FROM customer c
            LEFT JOIN (
                SELECT c_id, MAX(consult_date) AS max_consult_date
                FROM consultation_memo
                GROUP BY c_id
            ) m ON c.c_id = m.c_id
            WHERE m.max_consult_date IS NULL OR m.max_consult_date <= DATE_SUB(NOW(), INTERVAL 30 DAY)
            
            UNION ALL
            
            -- (3) 오늘 상담이 예정된 고객
            SELECT DISTINCT c_id, '오늘 상담 예약 확정 내방 예정' AS reason 
            FROM pb_schedule 
            WHERE category = '상담' AND DATE(execution_date) = CURDATE() AND c_id IS NOT NULL
            
            UNION ALL
            
            -- (4) 최근 3개월 내에 타행 거액 거래 내역이 있는 고객
            SELECT DISTINCT c_id, '최근 3개월 내 타행 거액 이출금(1천만 원 이상) 발생' AS reason 
            FROM customer_transaction 
            WHERE opp_bank_name != '우리은행' AND ct_type = 'W' AND amount >= 10000000 
              AND ct_datetime >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
              
            UNION ALL
            
            -- (5) 만기 예정 상품을 보유한 고객
            SELECT DISTINCT c_id, '30일 이내 만기 예정 금융 상품 보유' AS reason 
            FROM customer_product 
            WHERE expiration_date >= CURDATE() AND expiration_date <= DATE_ADD(CURDATE(), INTERVAL 30 DAY)
            
            UNION ALL
            
            -- (6) 고객 정보가 update된 이후에 AI 분석이 없었던 고객
            SELECT c_id, '고객 정보 업데이트 후 AI 분석 미수행' AS reason
            FROM customer
            WHERE update_time IS NOT NULL AND (analysis_time IS NULL OR update_time > analysis_time)
        ) r ON c.c_id = r.c_id
    """
    with get_db_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
        
        customer_map = {}
        for row in results:
            c_id = row["c_id"]
            if c_id not in customer_map:
                customer_map[c_id] = {
                    "c_id": c_id,
                    "name": row["name"],
                    "total_assets": row["total_assets"],
                    "net_worth": row["net_worth"],
                    "deposit": row["deposit"],
                    "loan": row["loan"],
                    "reasons": []
                }
            if row["reason"] not in customer_map[c_id]["reasons"]:
                customer_map[c_id]["reasons"].append(row["reason"])
                
        # 리스트 형태로 반환: [{"c_id": c_id, "name": name, ... "reasons": reasons}, ...]
        return list(customer_map.values())

def get_customer_ids_by_pb(u_id: str) -> list:
    """
    Retrieve list of customer IDs assigned to a specific PB (u_id) from the in_charge table.
    """
    query = """
        SELECT c_id 
        FROM in_charge 
        WHERE u_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (u_id,))
        results = cursor.fetchall()
        return [row["c_id"] for row in results]



