import wmill
from pymongo import MongoClient
from datetime import datetime, timedelta

def get_crawling_statistics(timeframe_days: int = 1) -> dict:
    client = MongoClient(wmill.get_variable("u/oudev2/mongo_uri_theanh"))
    db = client.financial_news

    start_date = datetime.utcnow() - timedelta(days=timeframe_days)

    try:
        #  crawl success
        total_crawled = db.raw_documents.count_documents(
            {"crawl_info.crawled_at": {"$gte": start_date}}
        )

        # extract success
        total_extracted = db.news_articles.count_documents(
            {"created_at": {"$gte": start_date}}
        )

        success_rate = (
            round((total_extracted / total_crawled * 100), 2)
            if total_crawled > 0
            else 0
        )

        # Khung giờ crawl nhiều nhất
        hourly_stats = list(
            db.raw_documents.aggregate(
                [
                    {
                        "$match": {"crawl_info.crawled_at": {"$gte": start_date}}
                    },  # select
                    {
                        "$group": {
                            "_id": {"$hour": "$crawl_info.crawled_at"},
                            "count": {"$sum": 1},
                        }
                    },
                    {"$sort": {"count": -1}},
                    {"$limit": 1},
                ]
            )
        )

        # Domain có nhiều bài crawl nhất
        domain_stats = list(
            db.raw_documents.aggregate(
                [
                    {"$match": {"crawl_info.crawled_at": {"$gte": start_date}}},
                    {"$group": {"_id": "$source.domain", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 1},
                ]
            )
        )

        # Kết quả
        peak_hour = hourly_stats[0] if hourly_stats else None
        top_domain = domain_stats[0] if domain_stats else None

        result = {
            "summary": {
                "total_crawled": total_crawled,
                "total_extracted": total_extracted,
                "success_rate": success_rate,
            },
            "peak_hour": {
                "hour": f"{peak_hour['_id']}:00" if peak_hour else "Không có dữ liệu",
                "count": peak_hour["count"] if peak_hour else 0,
            },
            "top_domain": {
                "domain": top_domain["_id"] if top_domain else "Không có dữ liệu",
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

        print(f"Khung giờ crawl nhiều nhất (UTC): {stats['peak_hour']['hour']}")
        print(f"Số bài: {stats['peak_hour']['count']} bài")
        print()

        print(f"Domain có nhiều bài nhất: {stats['top_domain']['domain']}")
        print(f"Số bài: {stats['top_domain']['count']} bài")

        return result
    else:
        print(f"Lỗi: {result['error']}")
        return result
