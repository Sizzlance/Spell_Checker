import re
from collections import defaultdict
from time import time, perf_counter
from heapq import nlargest
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def build_ngram(word, n=2):
    word = f"^{word}$"
    return [word[i:i + n] for i in range(len(word) - n + 1)]


def load_dictionary(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            words = set(word.strip().lower() for word in file if word.strip())
        return words
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='cp1251') as file:
            words = set(word.strip().lower() for word in file if word.strip())
        return words


def build_ngram_index(words, n=2):
    ngram_index = defaultdict(list)
    for word in words:
        for ngram in build_ngram(word, n):
            ngram_index[ngram].append(word)
    return ngram_index


def levenshtein_distance(s1: str, s2: str, max_distance=3):
    if abs(len(s1) - len(s2)) > max_distance:
        return max_distance + 1

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_similar_words(query: str, ngram_index: dict, dictionary_words, top_n=7):
    query_lower = query.lower()
    if query_lower in dictionary_words:
        return []

    query_ngrams = build_ngram(query_lower, n=2)
    candidate_words = set()

    for ngram in query_ngrams:
        candidate_words.update(ngram_index.get(ngram, set()))

    max_len_diff = 2
    candidate_words = {
        w for w in candidate_words
        if abs(len(w) - len(query_lower)) <= max_len_diff
    }

    if len(candidate_words) < top_n * 3:
        additional = {
            w for w in dictionary_words
            if abs(len(w) - len(query_lower)) <= 3
               and w[0] == query_lower[0]
        }
        candidate_words.update(additional)

    batch_size = max(10, min(50, len(candidate_words) // 4))
    candidate_list = list(candidate_words)

    word_scores = defaultdict(float)
    lock = Lock()

    def process_batch(batch):
        batch_scores = dict()
        for word in batch:
            score = 0

            if word[0] == query_lower[0]:
                score += 0.5
            if word[-1] == query_lower[-1]:
                score += 3.0

            length_diff = abs(len(word) - len(query_lower))
            score += length_diff * -0.5

            if score >= 2.0:
                distance = levenshtein_distance(query_lower, word)
                score += distance * -0.5

            batch_scores[word] = score
        return batch_scores

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for i in range(0, len(candidate_list), batch_size):
            batch = candidate_list[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch))

        for future in as_completed(futures):
            batch_scores = future.result()
            with lock:
                for word, score in batch_scores.items():
                    word_scores[word] = score

    top_words = nlargest(top_n * 2, word_scores.items(), key=lambda item: item[1])

    result = []
    seen = set()
    for word, score in top_words:
        if (word not in seen and
                word != query_lower and
                abs(len(word) - len(query_lower)) <= 2):
            seen.add(word)
            result.append(word)
            if len(result) >= top_n:
                break

    return result[:top_n]


def check_text_spelling(text, ngram_index, dictionary_words):
    words_to_check = extract_words(text)
    if not words_to_check:
        return

    # Первая фаза: быстрая проверка слов в словаре
    start_time = perf_counter()
    correct_words = sum(1 for word in words_to_check if word.lower() in dictionary_words)

    # Вторая фаза: обработка только ошибочных слов
    incorrect_words = [word for word in words_to_check if word.lower() not in dictionary_words]

    if not incorrect_words:
        elapsed = perf_counter() - start_time
        accuracy = (correct_words / len(words_to_check)) * 100
        print(f"\nВсе слова правильные. Время проверки: {elapsed:.3f} сек")
        return

    # Параллельная обработка ошибочных слов
    print(f"\nНайдено {len(incorrect_words)} потенциальных ошибок:")
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_word = {
            executor.submit(find_similar_words, word, ngram_index, dictionary_words): word
            for word in incorrect_words
        }

        for i, future in enumerate(as_completed(future_to_word), 1):
            word = future_to_word[future]
            try:
                suggestions = future.result()
                elapsed = perf_counter() - start_time
                print(f"\n{i}/{len(incorrect_words)}. Слово '{word}':", end=' ')

                if suggestions:
                    print(f"Возможные исправления ({elapsed:.2f} сек):")
                    for j, suggestion in enumerate(suggestions, 1):
                        print(f"  {j}. {suggestion.capitalize()}")
                else:
                    print("Нет подходящих вариантов.")
            except Exception as e:
                print(f"\nОшибка при обработке слова '{word}': {e}")

    elapsed = perf_counter() - start_time
    accuracy = (correct_words / len(words_to_check)) * 100
    print(f"\nРезультат: {correct_words}/{len(words_to_check)} ({accuracy:.1f}%) правильных слов")
    print(f"Общее время проверки: {elapsed:.2f} сек")

def extract_words(text):
    words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text)
    return words


def main():
    print("Загрузка словаря...")
    start_load = time()
    dictionary_words = load_dictionary("russian.txt")
    print(f"Словарь загружен: {len(dictionary_words)} слов")

    print("Построение ngram индекса...")
    ngram_index = build_ngram_index(dictionary_words)
    print(f"Индекс построен: {len(ngram_index)} ngrams")
    print(f"Время загрузки: {time() - start_load:.2f} сек\n")

    while True:
        try:
            print("Введите текст для проверки орфографии (или 'выход' для завершения):")
            user_text = input().strip()

            if user_text.lower() in ('выход', 'exit', 'quit'):
                break

            if not user_text:
                continue

            check_text_spelling(user_text, ngram_index, dictionary_words)

        except KeyboardInterrupt:
            return

if __name__ == "__main__":
    main()