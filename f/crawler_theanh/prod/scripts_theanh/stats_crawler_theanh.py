import wmill
from pymongo import MongoClient
from datetime import datetime, timedelta


def get_detailed_llm_stats_for_peak_hour(peak_hour_utc: int) -> dict:
    """
    Thống kê chi tiết LLM requests trong khung giờ cao điểm
    peak_hour_utc: giờ UTC có crawl nhiều nhất (ví dụ: 4)
    """
    client = MongoClient(wmill.get_variable("u/oudev2/mongo_uri_theanh"))
    db = client.financial_news
    
    # Tạo khung giờ 1 tiếng: từ 3h30 đến 4h30 UTC
    today = datetime.utcnow().date()
    start_time = datetime.combine(today, datetime.min.time()).replace(
        hour=peak_hour_utc - 1, minute=30, second=0, microsecond=0
    )
    end_time = datetime.combine(today, datetime.min.time()).replace(
        hour=peak_hour_utc, minute=30, second=0, microsecond=0
    )
    
    # Lấy tất cả LLM requests trong khung giờ này
    llm_requests = list(db.news_articles.aggregate([
        {
            "$match": {
                "created_at": {
                    "$gte": start_time,
                    "$lt": end_time
                }
            }
        },
        {
            "$project": {
                "hour": {"$hour": "$created_at"},
                "minute": {"$minute": "$created_at"},
                "five_minute_block": {
                    "$floor": {"$divide": [{"$minute": "$created_at"}, 5]}
                },
                "tokens": "$extraction.token_usage.total_tokens",
                "prompt_tokens": "$extraction.token_usage.prompt_tokens",
                "completion_tokens": "$extraction.token_usage.completion_tokens"
            }
        }
    ]))
    
    # Thống kê theo 5 phút
    five_min_stats = {}
    for req in llm_requests:
        hour_int = int(req['hour']) if req['hour'] is not None else 0
        minute_block = int(req['five_minute_block']) if req['five_minute_block'] is not None else 0
        
        block_key = f"{hour_int:02d}:{minute_block * 5:02d}-{(minute_block + 1) * 5:02d}"
        
        if block_key not in five_min_stats:
            five_min_stats[block_key] = {
                "request_count": 0,
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        
        five_min_stats[block_key]["request_count"] += 1
        five_min_stats[block_key]["total_tokens"] += req.get("tokens", 0) or 0
        five_min_stats[block_key]["total_prompt_tokens"] += req.get("prompt_tokens", 0) or 0
        five_min_stats[block_key]["total_completion_tokens"] += req.get("completion_tokens", 0) or 0
    
    # Tính tổng cho cả khung giờ
    total_requests = len(llm_requests)
    total_tokens = sum((req.get("tokens", 0) or 0) for req in llm_requests)
    total_prompt_tokens = sum((req.get("prompt_tokens", 0) or 0) for req in llm_requests)
    total_completion_tokens = sum((req.get("completion_tokens", 0) or 0) for req in llm_requests)
    
    # Tính requests per minute (trung bình)
    duration_minutes = 60
    avg_requests_per_minute = total_requests / duration_minutes if duration_minutes > 0 else 0
    
    return {
        "time_range": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_minutes": duration_minutes
        },
        "summary": {
            "total_requests": total_requests,
            "avg_requests_per_minute": round(avg_requests_per_minute, 2),
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "avg_tokens_per_request": round(total_tokens / total_requests, 2) if total_requests > 0 else 0
        },
        "five_minute_breakdown": five_min_stats
    }


def get_llm_statistics_from_db(timeframe_hours: int = 1) -> dict:
    """
    Lấy thống kê LLM requests từ database thay vì import
    """
    client = MongoClient(wmill.get_variable("u/oudev2/mongo_uri_theanh"))
    db = client.financial_news
    
    start_date = datetime.utcnow() - timedelta(hours=timeframe_hours)
    
    # Đếm tổng số bài đã extract (successful LLM calls)
    total_extracted = db.news_articles.count_documents({
        "created_at": {"$gte": start_date}
    })
    
    # Đếm số lỗi LLM
    llm_errors = db.errors.count_documents({
        "timestamp": {"$gte": start_date},
        "source": "llm_extraction"
    })
    
    # Tính requests theo giờ từ news_articles
    hourly_stats = list(db.news_articles.aggregate([
        {"$match": {"created_at": {"$gte": start_date}}},
        {"$group": {
            "_id": {"$hour": "$created_at"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]))
    
    # Chuyển thành dict dễ đọc
    requests_by_hour = {}
    for hour in hourly_stats:
        hour_id = hour['_id']
        if hour_id is not None:
            hour_key = f"{int(hour_id)}:00"
            requests_by_hour[hour_key] = int(hour['count'])
    
    total_requests = total_extracted + llm_errors
    
    return {
        "total_requests": int(total_requests),
        "successful_requests": int(total_extracted),
        "failed_requests": int(llm_errors),
        "requests_by_hour": requests_by_hour,
        "success_rate": float(
            (total_extracted / total_requests * 100) 
            if total_requests > 0 else 100
        )
    }


def get_crawling_statistics(timeframe_days: int = 1) -> dict:
    client = MongoClient(wmill.get_variable("u/oudev2/mongo_uri_theanh"))
    db = client.financial_news

    start_date = datetime.utcnow() - timedelta(days=timeframe_days)

    try:
        # Crawl success
        total_crawled = db.raw_documents.count_documents(
            {"crawl_info.crawled_at": {"$gte": start_date}}
        )

        # Extract success
        total_extracted = db.news_articles.count_documents(
            {"created_at": {"$gte": start_date}}
        )

        success_rate = float(
            round((total_extracted / total_crawled * 100), 2)
            if total_crawled > 0 else 0
        )

        # Error statistics
        error_stats = list(
            db.errors.aggregate([
                {"$match": {"timestamp": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$source",
                    "count": {"$sum": 1},
                    "types": {"$addToSet": "$error_type"}
                }},
                {"$sort": {"count": -1}}
            ])
        )

        # LLM statistics - dùng function mới
        llm_stats = get_llm_statistics_from_db(timeframe_hours=timeframe_days*24)

        # Peak hour for crawling - tìm giờ có crawl nhiều nhất
        hourly_stats = list(
            db.raw_documents.aggregate([
                {"$match": {"crawl_info.crawled_at": {"$gte": start_date}}},
                {"$group": {
                    "_id": {"$hour": "$crawl_info.crawled_at"},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 1}
            ])
        )

        peak_hour = None
        if hourly_stats and hourly_stats[0]['_id'] is not None:
            peak_hour = {
                "hour": int(hourly_stats[0]['_id']),
                "count": int(hourly_stats[0]['count'])
            }
        
        # Thống kê chi tiết LLM trong khung giờ cao điểm
        peak_hour_llm_stats = {}
        if peak_hour and peak_hour['hour'] is not None:
            peak_hour_llm_stats = get_detailed_llm_stats_for_peak_hour(peak_hour['hour'])

        # Top domain
        domain_stats = list(
            db.raw_documents.aggregate([
                {"$match": {"crawl_info.crawled_at": {"$gte": start_date}}},
                {"$group": {"_id": "$source.domain", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 1}
            ])
        )

        top_domain = None
        if domain_stats:
            top_domain = {
                "domain": str(domain_stats[0]['_id']) if domain_stats[0]['_id'] else "No data",
                "count": int(domain_stats[0]['count'])
            }

        result = {
            "summary": {
                "total_crawled": int(total_crawled),
                "total_extracted": int(total_extracted),
                "success_rate": success_rate,
            },
            "errors": error_stats,
            "llm_requests": llm_stats,
            "peak_hour": {
                "hour": f"{peak_hour['hour']}:00" if peak_hour else "No data",
                "count": peak_hour["count"] if peak_hour else 0,
            },
            "peak_hour_llm_detail": peak_hour_llm_stats,
            "top_domain": {
                "domain": top_domain["domain"] if top_domain else "No data",
                "count": top_domain["count"] if top_domain else 0,
            },
        }

        return {"success": True, "statistics": result}

    except Exception as e:
        return {"success": False, "error": str(e)}


def main(timeframe_days: int = 1) -> dict:
    result = get_crawling_statistics(timeframe_days)

    if result["success"]:
        stats = result["statistics"]

        print("THỐNG KÊ CRAWLING")
        print("=================")
        print(f"Khoảng thời gian: {timeframe_days} ngày")
        print(f"Tổng số bài đã crawl: {stats['summary']['total_crawled']}")
        print(f"Bài extract thành công: {stats['summary']['total_extracted']}")
        print(f"Tỉ lệ thành công: {stats['summary']['success_rate']}%")
        print()

        print("THỐNG KÊ LLM REQUESTS TỔNG QUAN")
        print("================================")
        llm = stats['llm_requests']
        print(f"Tổng requests: {llm['total_requests']}")
        print(f"Successful: {llm['successful_requests']}")
        print(f"Failed: {llm['failed_requests']}")
        print(f"Success rate: {llm['success_rate']:.1f}%")
        print(f"Requests theo giờ: {llm['requests_by_hour']}")
        print()

        # Hiển thị thống kê chi tiết khung giờ cao điểm
        if stats['peak_hour_llm_detail'] and stats['peak_hour']['hour'] != "No data":
            peak_detail = stats['peak_hour_llm_detail']
            peak_hour = stats['peak_hour']['hour']
            
            print(f"THỐNG KÊ CHI TIẾT LLM - KHUNG GIỜ CAO ĐIỂM ({peak_hour} UTC)")
            print("=" * 60)
            print(f"Khung giờ: {peak_detail['time_range']['start']} to {peak_detail['time_range']['end']}")
            print()
            
            summary = peak_detail['summary']
            print("TỔNG QUAN:")
            print(f"  • Tổng requests: {summary['total_requests']}")
            print(f"  • Requests trung bình/phút: {summary['avg_requests_per_minute']}")
            print(f"  • Tổng tokens: {summary['total_tokens']:,}")
            print(f"  • Tokens trung bình/request: {summary['avg_tokens_per_request']:,}")
            print(f"  • Prompt tokens: {summary['total_prompt_tokens']:,}")
            print(f"  • Completion tokens: {summary['total_completion_tokens']:,}")
            print()
            
            if peak_detail['five_minute_breakdown']:
                print("CHI TIẾT THEO 5 PHÚT:")
                for time_block, block_stats in peak_detail['five_minute_breakdown'].items():
                    print(f"  {time_block}:")
                    print(f"    • Requests: {block_stats['request_count']}")
                    print(f"    • Tokens: {block_stats['total_tokens']:,}")
                    if block_stats['request_count'] > 0:
                        avg_tokens = block_stats['total_tokens'] / block_stats['request_count']
                        print(f"    • Tokens/request: {avg_tokens:,.0f}")
                print()
            else:
                print("Không có dữ liệu chi tiết theo 5 phút")
                print()

        print("THỐNG KÊ LỖI")
        print("============")
        if stats['errors']:
            for error in stats['errors']:
                print(f"{error['_id']}: {error['count']} lỗi")
                print(f"  Loại lỗi: {', '.join(error['types'])}")
        else:
            print("Không có lỗi nào")
        print()

        print(f"Khung giờ crawl nhiều nhất (UTC): {stats['peak_hour']['hour']}")
        print(f"Số bài: {stats['peak_hour']['count']} bài")
        print()
        print(f"Domain có nhiều bài nhất: {stats['top_domain']['domain']}")
        print(f"Số bài: {stats['top_domain']['count']} bài")

        return result
    else:
        print(f"Lỗi: {result['error']}")
        return result