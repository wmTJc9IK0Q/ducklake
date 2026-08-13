//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_file_list_cache.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "common/index.hpp"
#include "storage/ducklake_metadata_info.hpp"

#include <list>

namespace duckdb {

//! Key identifying one DuckLakeMetadataManager::GetFilesForTable result.
//!
//! The file list a scan reads is a pure function of (table, snapshot,
//! filter pushdown) over committed metadata: the metadata rows a snapshot
//! references are immutable once that snapshot is committed, and the generated
//! SQL fully encodes the filter pushdown (zone maps, bucket pruning, stats CTEs).
//! `query` is the SQL before catalog/snapshot placeholder substitution, so two
//! executions that resolve the same snapshot and build the same filters produce
//! the same key.
struct DuckLakeFileListCacheKey {
	DuckLakeFileListCacheKey(TableIndex table_id_p, idx_t snapshot_id_p, idx_t schema_version_p, string query_p)
	    : table_id(table_id_p), snapshot_id(snapshot_id_p), schema_version(schema_version_p),
	      query(std::move(query_p)) {
	}

	TableIndex table_id;
	idx_t snapshot_id;
	idx_t schema_version;
	//! Generated metadata SQL before {METADATA_CATALOG}/{SNAPSHOT_ID} substitution.
	string query;

	bool operator==(const DuckLakeFileListCacheKey &other) const {
		return table_id.index == other.table_id.index && snapshot_id == other.snapshot_id &&
		       schema_version == other.schema_version && query == other.query;
	}
};

struct DuckLakeFileListCacheKeyHash {
	size_t operator()(const DuckLakeFileListCacheKey &key) const {
		hash_t result = duckdb::Hash(key.snapshot_id);
		result = duckdb::CombineHash(result, duckdb::Hash(key.schema_version));
		result = duckdb::CombineHash(result, duckdb::Hash(key.table_id));
		result = duckdb::CombineHash(result, duckdb::Hash(key.query.c_str(), key.query.size()));
		return result;
	}
};

//! A bounded LRU of DuckLakeMetadataManager::GetFilesForTable results, keyed by
//! (table, snapshot, generated query).
//!
//! Snapshot-keyed means a new commit - a new snapshot id - misses and
//! recomputes, so cached entries never serve data a later snapshot would not
//! have served: the correctness frontier is identical to recomputing every
//! time. This is what lets repeated identical scans skip the expensive
//! file-list/statistics metadata SQL entirely.
class DuckLakeFileListCache {
public:
	//! The cap on live entries. Metadata files rarely number past a few per
	//! table, so this is generous; arbitrary filter variation is what grows the
	//! key space, and the LRU keeps the working set of distinct filters.
	static constexpr idx_t DUCKLAKE_FILE_LIST_CACHE_CAPACITY = 512;

public:
	//! Returns the cached file list for (table, snapshot, query), or nullptr.
	shared_ptr<const vector<DuckLakeFileListEntry>> Lookup(const DuckLakeFileListCacheKey &key) {
		lock_guard<mutex> guard(lock);
		auto entry = entries.find(key);
		if (entry == entries.end()) {
			misses++;
			return nullptr;
		}
		hits++;
		// LRU: move the touched key to the front of the eviction order.
		lru.splice(lru.begin(), lru, entry->second.lru_it);
		return entry->second.value;
	}

	//! Store the file list for (table, snapshot, query). Misses and recomputes
	//! when the same key was already cached with different content are not
	//! possible: the key is deterministic for identical committed metadata.
	void Store(const DuckLakeFileListCacheKey &key, const vector<DuckLakeFileListEntry> &files) {
		lock_guard<mutex> guard(lock);
		auto entry = entries.find(key);
		if (entry != entries.end()) {
			// Refresh the existing entry; the value is a pure function of the
			// key, so keep the pre-existing shared result.
			lru.splice(lru.begin(), lru, entry->second.lru_it);
			return;
		}
		if (entries.size() >= DUCKLAKE_FILE_LIST_CACHE_CAPACITY) {
			EvictLocked();
		}
		lru.push_front(key);
		entries.emplace(key,
		                CacheEntry { make_shared_ptr<vector<DuckLakeFileListEntry>>(files), lru.begin()});
	}

	//! Drop every entry. Used by tests and by detach-time cleanup.
	void Clear() {
		lock_guard<mutex> guard(lock);
		entries.clear();
		lru.clear();
	}

	//! Hits and misses since construction (diagnostics only).
	uint64_t Hits() const {
		return hits;
	}
	uint64_t Misses() const {
		return misses;
	}

private:
	//! Evict the least-recently-used entry. Caller holds lock.
	void EvictLocked() {
		auto last = lru.back();
		lru.pop_back();
		entries.erase(last);
	}

private:
	struct CacheEntry {
		shared_ptr<const vector<DuckLakeFileListEntry>> value;
		std::list<DuckLakeFileListCacheKey>::iterator lru_it;
	};

	mutable mutex lock;
	unordered_map<DuckLakeFileListCacheKey, CacheEntry, DuckLakeFileListCacheKeyHash> entries;
	std::list<DuckLakeFileListCacheKey> lru;
	//! Diagnostics only - not part of any cache contract.
	atomic<uint64_t> hits {0};
	atomic<uint64_t> misses {0};
};

} // namespace duckdb
