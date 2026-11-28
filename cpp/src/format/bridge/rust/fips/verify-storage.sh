#!/usr/bin/env bash
set -euo pipefail

nm_tool=$1
storage=$2

"$nm_tool" --format=posix --defined-only "$storage" | awk '
    $1 == "FIPS_mode" || $1 == "awslc_version_string" {
        if ($2 == "t") {
            local_symbols[$1] = 1
        } else {
            print "Expected private AWS-LC FIPS symbol: " $1
            bad = 1
        }
    }
    $1 ~ /^(ring_core_|aws_lc_[0-9]+_)/ { print "Unexpected crypto symbol: " $1; bad = 1 }
    END {
        if (!local_symbols["FIPS_mode"] || !local_symbols["awslc_version_string"]) {
            print "Expected local AWS-LC FIPS symbols are missing"
            bad = 1
        }
        exit bad
    }
'
