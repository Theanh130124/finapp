import wmill
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime
import uuid
from typing import Dict, List
from openai import OpenAI
import json


client_openai = OpenAI(
    base_url=wmill.get_variable("f/variables_theanh/open_router_url"),
    api_key=wmill.get_variable("f/variables_theanh/open_ai_key"),
)

# Biến toàn cục để thống kê LLM requests
LLM_REQUEST_TRACKER = {
    "total_requests": 0,
    "requests_by_hour": {},
    "failed_requests": 0,
}


BATCH_EXTRACTION_PROMPT = """
You are a financial news extraction expert. Extract structured information from MULTIPLE news articles in batch.

**ARTICLES DATA:**
{articles_text}

**EXTRACTION INSTRUCTIONS:**

Extract and return a JSON object with the following structure:

{{
  "articles": [
    {{
      "original_url": "URL of the article",
      "content": {{
        "headline": "Main headline",
        "subheadline": "Subheadline if exists",
        "summary": "Brief 2-3 sentence summary",
        "body": "Full article text, cleaned",
        "author": "Author name if available"
      }},
      "source": {{
        "name": "Source name (Bloomberg, Reuters, etc.)",
        "credibility_score": 0.0-1.0
      }},
      "timing": {{
        "published_at": "ISO datetime",
        "market_session": "pre_market|market_hours|after_hours|closed"
      }},
      "classification": {{
        "primary_category": "earnings|m&a|guidance|product_launch|regulatory|management_change|economic_data|other",
        "sub_categories": ["list", "of", "subcategories"],
        "topics": ["list", "of", "topics"]
      }},
      "companies_mentioned": [
        {{
          "ticker": "AAPL",
          "company_name": "Apple Inc.",
          "relevance_score": 0.0-1.0,
          "mention_type": "primary_subject|secondary_subject|mentioned|compared_to",
          "sentiment": -1.0 to 1.0,
          "context": "Brief context of how company is mentioned"
        }}
      ],
      "events_extracted": [
        {{
          "event_type": "earnings_beat|earnings_miss|dividend_increase|stock_split|merger_announced|ceo_change|product_launch|guidance_raise|guidance_lower|other",
          "description": "Brief description",
          "companies_affected": ["AAPL", "MSFT"],
          "impact_magnitude": numeric value if applicable (e.g., % beat),
          "confidence": 0.0-1.0
        }}
      ],
      "sentiment": {{
        "overall_sentiment": -1.0 to 1.0,
        "sentiment_magnitude": 0.0-1.0,
        "emotional_tone": {{
          "fear": 0.0-1.0,
          "greed": 0.0-1.0,
          "optimism": 0.0-1.0,
          "pessimism": 0.0-1.0,
          "urgency": 0.0-1.0
        }},
        "market_impact_score": 0.0-1.0
      }},
      "confidence_score": 0.0-1.0
    }}
  ]
}}

**GUIDELINES:**

1. Return EXACTLY one JSON object with an "articles" array containing all extracted articles in the SAME ORDER as provided.
2. For each article, include the "original_url" field to match with source.
3. Process ALL articles in the batch - do not skip any.
4. Use the COMPLETE article text for accurate extraction - do not truncate or summarize the original content.
5. Follow the same extraction guidelines as single article processing.

Return ONLY the JSON object, no additional text.
"""


def track_llm_request(success: bool = True):
    """Theo dõi số lượng request đến LLM"""
    global LLM_REQUEST_TRACKER
    current_hour = datetime.utcnow().strftime("%Y-%m-%d %H:00")

    LLM_REQUEST_TRACKER["total_requests"] += 1

    if not success:
        LLM_REQUEST_TRACKER["failed_requests"] += 1

    if current_hour not in LLM_REQUEST_TRACKER["requests_by_hour"]:
        LLM_REQUEST_TRACKER["requests_by_hour"][current_hour] = 0
    LLM_REQUEST_TRACKER["requests_by_hour"][current_hour] += 1


def log_error(
    db,
    error_type: str,
    error_message: str,
    source: str,
    document_id: str = None,
    url: str = None,
    metadata: dict = None,
):
    """Ghi lỗi vào collection errors"""
    error_doc = {
        "_id": str(uuid.uuid4()),
        "error_type": error_type,
        "error_message": error_message,
        "source": source,
        "document_id": document_id,
        "url": url,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow(),
        "resolved": False,
    }
    return db.errors.insert_one(error_doc).inserted_id


def calculate_batch_size(articles_data: List[Dict]) -> int:
    """Tính toán kích thước batch dựa trên tổng độ dài văn bản"""
    total_chars = sum(len(article.get('text', '')) for article in articles_data)
    total_tokens_estimate = total_chars / 4  # Ước tính thô: 1 token ≈ 4 characters
    
    # Nếu ước tính vượt quá 100k tokens, giảm batch size hoặc cảnh báo
    if total_tokens_estimate > 100000:
        print(f"Warning: Large batch detected. Estimated tokens: {total_tokens_estimate:.0f}")
    
    return len(articles_data)  # Giữ nguyên batch size, để model tự xử lý


def batch_extract_news_with_llm(
    articles_data: List[Dict], model: str = "openai/gpt-oss-20b:free", max_retries: int = 3
) -> Dict:
    """Extract thông tin từ nhiều bài viết cùng lúc với retry logic - KHÔNG giới hạn nội dung"""
    
    for attempt in range(max_retries):
        try:
            # Chuẩn bị dữ liệu articles cho prompt - TOÀN BỘ nội dung
            articles_text = ""
            for i, article in enumerate(articles_data):
                articles_text += f"\n--- ARTICLE {i + 1} ---\n"
                articles_text += f"URL: {article['url']}\n"
                articles_text += f"Title: {article.get('title', '')}\n"
                articles_text += f"Publication Date: {article.get('publication_date', '')}\n"
                articles_text += f"Content: {article['text']}\n"  # TOÀN BỘ nội dung, không giới hạn

            prompt = BATCH_EXTRACTION_PROMPT.format(articles_text=articles_text)

            # Tính toán tổng độ dài để log
            total_chars = len(prompt)
            print(f"Batch extraction attempt {attempt + 1}: {total_chars:,} characters, {len(articles_data)} articles")

            response = client_openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial news extraction expert. Extract structured information from multiple articles in batch. Use the COMPLETE article text for accurate extraction. Return a JSON object with 'articles' array in the same order as input.",
                    },
                    {"role": "user", "content": prompt},
                ],
                extra_headers={
                    "HTTP-Referer": "https://windmill.pythera.ai",
                    "X-Title": "Batch Extract with LLM By TheAnh",
                },
                temperature=0.1,
                max_tokens=128000,  # Để max tokens cho model xử lý batch lớn
                response_format={"type": "json_object"},
            )

            extracted_data = json.loads(response.choices[0].message.content)

            # Validate response structure
            if "articles" not in extracted_data:
                raise ValueError("LLM response missing 'articles' array")
            
            if len(extracted_data["articles"]) != len(articles_data):
                raise ValueError(f"Expected {len(articles_data)} articles, got {len(extracted_data['articles'])}")

            # Validate URLs để đảm bảo khớp với input
            for i, (input_article, output_article) in enumerate(zip(articles_data, extracted_data["articles"])):
                output_url = output_article.get("original_url", "")
                input_url = input_article["url"]
                if output_url and output_url != input_url:
                    print(f"Warning: URL mismatch at index {i}: {input_url} vs {output_url}")

            # Lấy thông tin token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            print(f"Batch extraction successful: {token_usage['total_tokens']:,} tokens used")

            track_llm_request(success=True)

            return {
                "extracted": extracted_data,
                "usage": token_usage,
                "model": model,
            }

        except Exception as e:
            print(f"Batch extraction attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                track_llm_request(success=False)
                raise e
            # Chờ một chút trước khi retry
            import time
            time.sleep(2)


def main(batch_size: int = 12, max_retries: int = 3) -> dict:
    """
    Batch extract nhiều bài viết cùng lúc

    Args:
        batch_size: Số lượng bài viết để extract cùng lúc (mặc định: 12)
        max_retries: Số lần thử lại tối đa cho batch extraction (mặc định: 3)
    """
    client = MongoClient(wmill.get_variable("u/oudev2/mongo_uri_theanh"))
    db = client.financial_news

    try:
        # Tìm các bài viết chưa được extract, ưu tiên những bài có retry_count thấp
        pending_articles = list(
            db.raw_documents.find({
                "$or": [
                    {"processing_status.status": "pending"},
                    {"processing_status.status": "failed", "processing_status.retry_count": {"$lt": max_retries}}
                ]
            }).sort([("processing_status.retry_count", 1), ("metadata.publication_date", -1)]).limit(batch_size)
        )

        if not pending_articles:
            return {
                "success": True,
                "message": "No pending articles found",
                "processed_count": 0,
            }

        print(f"Found {len(pending_articles)} pending articles for batch processing")

        # Chuẩn bị dữ liệu cho batch extraction - TOÀN BỘ nội dung
        articles_data = []
        total_chars = 0
        
        for article in pending_articles:
            article_text = article["content"]["text"]
            articles_data.append(
                {
                    "document_id": article["_id"],
                    "url": article["source"]["url"],
                    "title": article["metadata"].get("title", ""),
                    "publication_date": article["metadata"].get("publication_date", ""),
                    "text": article_text,  # TOÀN BỘ nội dung
                }
            )
            total_chars += len(article_text)

        print(f"Total characters in batch: {total_chars:,} (approx {total_chars/4:.0f} tokens)")

        # Cập nhật trạng thái đang xử lý và tăng retry count
        document_ids = [article["_id"] for article in pending_articles]
        current_retry_counts = {article["_id"]: article["processing_status"].get("retry_count", 0) 
                               for article in pending_articles}
        
        db.raw_documents.update_many(
            {"_id": {"$in": document_ids}},
            {
                "$set": {"processing_status.status": "processing"},
                "$inc": {"processing_status.retry_count": 1}
            },
        )

        # Thực hiện batch extraction với retry
        result = batch_extract_news_with_llm(articles_data, max_retries=max_retries)

        extracted_articles = result["extracted"].get("articles", [])

        # Xử lý kết quả và lưu vào database
        processed_results = []
        success_count = 0

        for i, (original_article, extracted_article) in enumerate(
            zip(pending_articles, extracted_articles)
        ):
            try:
                # Kiểm tra URL khớp (có thể bỏ qua nếu không khớp nhưng log lại)
                original_url = original_article["source"]["url"]
                extracted_url = extracted_article.get("original_url", "")
                if extracted_url and extracted_url != original_url:
                    print(f"Warning: URL mismatch for article {i + 1}: {original_url} vs {extracted_url}")

                # Tạo document cho news_articles
                article_doc = {
                    "_id": str(uuid.uuid4()),
                    "raw_document_id": original_article["_id"],
                    "content": extracted_article.get("content", {}),
                    "source": extracted_article.get("source", {}),
                    "timing": extracted_article.get("timing", {}),
                    "classification": extracted_article.get("classification", {}),
                    "companies_mentioned": extracted_article.get("companies_mentioned", []),
                    "events_extracted": extracted_article.get("events_extracted", []),
                    "sentiment": extracted_article.get("sentiment", {}),
                    "extraction": {
                        "model": result["model"],
                        "extracted_at": datetime.utcnow(),
                        "confidence_score": extracted_article.get("confidence_score", 0.0),
                        "token_usage": result["usage"],
                        "batch_processed": True,
                        "batch_size": len(pending_articles),
                    },
                    "created_at": datetime.utcnow(),
                }

                db.news_articles.insert_one(article_doc)

                # Cập nhật trạng thái raw document - thành công
                db.raw_documents.update_one(
                    {"_id": original_article["_id"]},
                    {
                        "$set": {
                            "processing_status.status": "completed",
                            "processing_status.processed_at": datetime.utcnow(),
                            "processing_status.last_error": None,
                        }
                    },
                )

                processed_results.append(
                    {
                        "success": True,
                        "article_id": article_doc["_id"],
                        "document_id": original_article["_id"],
                        "companies_found": len(extracted_article.get("companies_mentioned", [])),
                        "sentiment": extracted_article.get("sentiment", {}).get("overall_sentiment"),
                        "confidence_score": extracted_article.get("confidence_score", 0.0),
                    }
                )
                success_count += 1

            except Exception as e:
                # Log lỗi cho bài viết cụ thể
                error_id = log_error(
                    db,
                    error_type="batch_article_processing_error",
                    error_message=str(e),
                    source="batch_extraction",
                    document_id=original_article["_id"],
                    url=original_article["source"]["url"],
                    metadata={
                        "retry_count": current_retry_counts[original_article["_id"]],
                        "batch_index": i
                    }
                )

                processed_results.append(
                    {
                        "success": False,
                        "document_id": original_article["_id"],
                        "error": str(e),
                        "error_id": error_id,
                    }
                )

                # Đánh dấu bài viết thất bại nhưng vẫn giữ retry_count
                db.raw_documents.update_one(
                    {"_id": original_article["_id"]},
                    {
                        "$set": {
                            "processing_status.status": "failed",
                            "processing_status.error_message": str(e),
                            "processing_status.error_id": error_id,
                            "processing_status.last_error": str(e),
                        }
                    },
                )

        print(f"Batch processing completed: {success_count}/{len(pending_articles)} successful")

        return {
            "success": True,
            "batch_processing": True,
            "total_processed": len(pending_articles),
            "successful_extractions": success_count,
            "failed_extractions": len(pending_articles) - success_count,
            "token_usage": result["usage"],
            "total_characters_processed": total_chars,
            "detailed_results": processed_results,
        }

    except Exception as e:
        # Log lỗi batch
        error_id = log_error(
            db,
            error_type="batch_extraction_error",
            error_message=str(e),
            source="batch_extraction_main",
            metadata={"batch_size": batch_size, "max_retries": max_retries},
        )

        print(f"Batch processing failed: {str(e)}")

        return {
            "success": False,
            "error": str(e),
            "error_id": error_id,
            "batch_processing": True,
        }