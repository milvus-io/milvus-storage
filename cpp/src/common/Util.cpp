#include "milvus-storage/common/Util.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <regex>
#include <sstream>
#include <ctime>
#include <iomanip>

namespace milvus_storage {
std::string
NormalizeToUtcZ(const std::string& s) {
    if (!s.empty() && s.back() == 'Z') {
        return s;
    }
    static const std::regex re(
        R"(^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})([+-])(\d{2}):(\d{2})$)");
    std::smatch m;
    if (!std::regex_match(s, m, re)) {
        throw std::runtime_error("Invalid RFC3339 time: " + s);
    }
    const std::string datetime = m[1];
    const char sign = m[2].str()[0];
    const int off_h = std::stoi(m[3]);
    const int off_m = std::stoi(m[4]);
    std::tm tm{};
    std::istringstream ss(datetime);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        throw std::runtime_error("Invalid datetime body: " + datetime);
    }
    time_t t = timegm(&tm);
    const int offset_sec = off_h * 3600 + off_m * 60;
    if (sign == '+') {
        t -= offset_sec;
    } else {
        t += offset_sec;
    }
    std::tm* utc = gmtime(&t);
    char buf[32];
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", utc) == 0) {
        throw std::runtime_error("strftime failed");
    }
    return std::string(buf);
}

}  // namespace milvus_storage
