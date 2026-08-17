#pragma once

#include "rfanns/types.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rfanns::cli
{
    class Arguments
    {
    public:
        Arguments(int argc, char **argv)
        {
            for (int index = 1; index < argc; index += 2)
            {
                const std::string name = argv[index];
                if (name.rfind("--", 0) != 0)
                    throw Error("expected an option name, got " + name);
                if (index + 1 >= argc)
                    throw Error("missing value for " + name);
                if (!values_.emplace(name.substr(2), argv[index + 1]).second)
                    throw Error("option was provided more than once: " + name);
            }
        }

        bool has(const std::string &name) const { return values_.count(name) != 0; }

        std::string get(const std::string &name, const std::string &fallback = {}) const
        {
            const auto found = values_.find(name);
            return found == values_.end() ? fallback : found->second;
        }

        std::string require(const std::string &name) const
        {
            const std::string value = get(name);
            if (value.empty())
                throw Error("--" + name + " is required");
            return value;
        }

        std::uint32_t u32(const std::string &name, std::uint32_t fallback) const
        {
            return has(name) ? static_cast<std::uint32_t>(std::stoul(get(name))) : fallback;
        }

        std::uint64_t u64(const std::string &name, std::uint64_t fallback) const
        {
            return has(name) ? static_cast<std::uint64_t>(std::stoull(get(name))) : fallback;
        }

        double real(const std::string &name, double fallback) const
        {
            return has(name) ? std::stod(get(name)) : fallback;
        }

        bool boolean(const std::string &name, bool fallback) const
        {
            if (!has(name))
                return fallback;
            std::string value = get(name);
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (value == "1" || value == "true" || value == "on")
                return true;
            if (value == "0" || value == "false" || value == "off")
                return false;
            throw Error("--" + name + " must be 0/1, true/false, or on/off");
        }

        void allow_only(std::initializer_list<const char *> names) const
        {
            std::unordered_set<std::string> allowed;
            for (const char *name : names)
                allowed.emplace(name);
            for (const auto &item : values_)
            {
                if (!allowed.count(item.first))
                    throw Error("unknown or intentionally removed option --" + item.first);
            }
        }

    private:
        std::unordered_map<std::string, std::string> values_;
    };

    inline std::vector<std::uint32_t> parse_u32_list(const std::string &text,
                                                     const std::string &option)
    {
        std::vector<std::uint32_t> result;
        std::stringstream stream(text);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            if (!token.empty())
                result.push_back(static_cast<std::uint32_t>(std::stoul(token)));
        }
        if (result.empty())
            throw Error("--" + option + " must contain at least one positive value");
        if (std::find(result.begin(), result.end(), 0) != result.end())
            throw Error("--" + option + " values must be positive");
        return result;
    }
}
