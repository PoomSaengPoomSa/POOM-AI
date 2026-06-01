from .db import get_db_cursor

def get_customer_features(customer_id: int, months: int = 3):
    """
    Get customer features extracted from the database for the given period (months).
    """
    query = """
        SELECT ci_id, category, contents, created_date
        FROM customer_information
        WHERE c_id = %s AND created_date >= DATE_SUB(NOW(), INTERVAL %s MONTH)
        ORDER BY created_date DESC
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id, months))
        results = cursor.fetchall()
        return results

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

def save_customer_feature(customer_id: int, category: str, contents: str):
    """
    Insert a new customer feature row into customer_information.
    """
    query = """
        INSERT INTO customer_information (c_id, category, contents, created_date)
        VALUES (%s, %s, %s, NOW())
    """
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, (customer_id, category, contents))
        return rows_affected > 0

def update_customer_feature(ci_id: int, contents: str):
    """
    Update the contents of a specific customer feature row in customer_information.
    """
    query = "UPDATE customer_information SET contents = %s, created_date = NOW() WHERE ci_id = %s"
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, (contents, ci_id))
        return rows_affected > 0

def get_customer_relationships_all(customer_id: int):
    """
    Retrieve all relationships including cr_id and information from customer_relationship table.
    """
    query = """
        SELECT cr_id, c_id, relationship, information, birthday, job, is_spouse, wedding_date
        FROM customer_relationship
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        results = cursor.fetchall()
        return results

def save_customer_relationship(customer_id: int, relationship: str, information: str, birthday: str = None, job: str = None, is_spouse: int = 0, wedding_date: str = None):
    """
    Insert a new relationship record into customer_relationship.
    """
    query = """
        INSERT INTO customer_relationship (c_id, relationship, information, birthday, job, is_spouse, wedding_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, (customer_id, relationship, information, birthday, job, is_spouse, wedding_date))
        return rows_affected > 0

def update_customer_relationship(cr_id: int, information: str, birthday: str = None, job: str = None, is_spouse: int = None, wedding_date: str = None):
    """
    Update an existing relationship record in customer_relationship.
    Only updates columns that are provided.
    """
    update_fields = []
    params = []
    
    update_fields.append("information = %s")
    params.append(information)
    
    if birthday is not None:
        update_fields.append("birthday = %s")
        params.append(birthday)
    if job is not None:
        update_fields.append("job = %s")
        params.append(job)
    if is_spouse is not None:
        update_fields.append("is_spouse = %s")
        params.append(is_spouse)
    if wedding_date is not None:
        update_fields.append("wedding_date = %s")
        params.append(wedding_date)
        
    params.append(cr_id)
    query = f"UPDATE customer_relationship SET {', '.join(update_fields)} WHERE cr_id = %s"
    
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, params)
        return rows_affected > 0

def save_customer_keyword_features(customer_id: int, keywords_str: str):
    """
    Update the customer table's features column with a comma-separated keywords string.
    """
    query = """
        UPDATE customer
        SET features = %s
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, (keywords_str, customer_id))
        return rows_affected > 0
