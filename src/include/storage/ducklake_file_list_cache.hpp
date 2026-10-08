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
//! The file list for a snapshot is NOT immutable once that snapshot is committed:
//! flushing inlined data moves rows the old snapshot saw out of the inlined tables
//! and into new data files that carry the old snapshot range. A file list cached
//! before such a commit, combined with inlined tables read after it, loses those
//! rows. So the key carries the snapshot the reading transaction started from
//! (`transaction_snapshot_id`) besides the one it scans (`snapshot_id`, which
//! differs under AT (VERSION => ...)). A transaction reads its metadata as of its
//! own snapshot, so equal keys see equal metadata, and any commit moves the
//! transaction snapshot of later readers and misses.
//!
//! `query` is the SQL before catalog/snapshot placeholder substitution and fully
//! encodes the filter pushdown (zone maps, bucket pruning, stats CTEs).
struct DuckLakeFileListCacheKey {
	DuckLakeFileListCacheKey(TableIndex table_id_p, idx_t transaction_snapshot_id_p, idx_t snapshot_id_p,
	                         idx_t schema_version_p, string query_p)
	    : table_id(table_id_p), transaction_snapshot_id(transaction_snapshot_id_p), snapshot_id(snapshot_id_p),
	      schema_version(schema_version_p), query(std::move(query_p)) {
	}

	TableIndex table_id;
	//! The snapshot of the reading transaction - the metadata state the list was computed from
	idx_t transaction_snapshot_id;
	//! The snapshot being scanned
	idx_t snapshot_id;
	idx_t schema_version;
	//! Generated metadata SQL before {METADATA_CATALOG}/{SNAPSHOT_ID} substitution.
	string query;

	bool operator==(const DuckLakeFileListCacheKey &other) const {
		return table_id.index == other.table_id.index && transaction_snapshot_id == other.transaction_snapshot_id &&
		       snapshot_id == other.snapshot_id && schema_version == other.schema_version && query == other.query;
	}
};

struct DuckLakeFileListCacheKeyHash {
	size_t operator()(const DuckLakeFileListCacheKey &key) const {
		hash_t result = duckdb::Hash(key.snapshot_id);
		result = duckdb::CombineHash(result, duckdb::Hash(key.transaction_snapshot_id));
		result = duckdb::CombineHash(result, duckdb::Hash(key.schema_version));
		result = duckdb::CombineHash(result, duckdb::Hash(key.table_id));
		result = duckdb::CombineHash(result, duckdb::Hash(key.query.c_str(), key.query.size()));
		return result;
	}
};

//! A bounded LRU of DuckLakeMetadataManager::GetFilesForTable results, keyed by
//! (table, transaction snapshot, scanned snapshot, generated query).
//!
//! Any commit moves the snapshot later transactions start from, so they miss and
//! recompute: cached entries never serve a file list the current metadata would
//! not produce. This is what lets repeated identical scans skip the expensive
//! file-list/statistics metadata SQL entirely.
class DuckLakeFileListCache {
public:
	//! The cap on live entries. Metadata files rarely number past a few per
	//! table, so this is generous; arbitrary filter variation is what grows the
	//! key space, and the LRU keeps the working set of distinct filters.
	static constexpr idx_t DUCKLAKE_FILE_LIST_CACHE_CAPACITY = 512;

public:
	//! Returns the cached file list for the key, or nullptr.
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

	//! Store the file list for the key. The same key cannot map to different
	//! content: equal keys read equal committed metadata.
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
