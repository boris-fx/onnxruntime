#pragma once

#include "scan_defs.h"

// The shader headers point to bytecode produced using "GeneratedShaders/GenerateShaders.bat"
#include "scan_clear_buffer_hlsl.h"
#include "scan_prescan_hlsl.h"
#include "scan_add_block_sums_hlsl.h"

class PrefixSum {
public:

    typedef int32_t T;

    ComPtr<ID3D12Device> m_device;

    std::shared_ptr<ComputeShader> m_shader_clear_buffer;
    std::shared_ptr<ComputeShader> m_shader_prescan;
    std::shared_ptr<ComputeShader> m_shader_add_block_sums;

    // pre-allocate intermediate buffers based on num elements!
    std::shared_ptr<ComputeBuffer> m_dummy_blocks_sums;
    std::unordered_map<size_t, std::shared_ptr<ComputeBuffer>> m_block_sums;
    std::unordered_map<size_t, std::shared_ptr<ComputeBuffer>> m_in_block_sums;

    explicit PrefixSum(ComPtr<ID3D12Device> device, unsigned int n_els): m_device(device), m_n_els(n_els) {
        m_shader_clear_buffer.reset(new ComputeShader(device, scan_clear_buffer_hlsl::cfg));
        m_shader_prescan.reset(new ComputeShader(device, scan_prescan_hlsl::cfg));
        m_shader_add_block_sums.reset(new ComputeShader(device, scan_add_block_sums_hlsl::cfg));

        m_dummy_blocks_sums.reset(new ComputeBuffer(device, sizeof(int32_t)));

        unsigned int n = n_els;
        while (true) {
            n = get_grid_sz(n);
            m_block_sums[n].reset(new ComputeBuffer(device, n * sizeof(int32_t)));
            m_in_block_sums[n].reset(new ComputeBuffer(device, n * sizeof(int32_t)));
            if (n == 1) break;
        }
    }

    void run(ComPtr<ID3D12GraphicsCommandList> commandList, ComPtr<ID3D12Resource> d_out, ComPtr<ID3D12Resource> d_in) {
        run_(commandList, d_out, d_in, m_n_els);
    }

private:
    unsigned int get_grid_sz(const unsigned int n) {
        return (n + MAX_BLOCK_SZ - 1) / MAX_BLOCK_SZ; // ceil division;
    }

    // really a recursive implementation..
    void run_(ComPtr<ID3D12GraphicsCommandList> commandList, ComPtr<ID3D12Resource> d_out, ComPtr<ID3D12Resource> d_in, const unsigned int numElems) {

        clearBuffer(commandList, d_out, numElems);

        unsigned int grid_sz = get_grid_sz(numElems);

        auto d_block_sums = m_block_sums.at(grid_sz);
        clearBuffer(commandList, d_block_sums->resource(), grid_sz);

        gpuPrescan(commandList, d_out, d_in, d_block_sums->resource(), grid_sz, numElems);

        if (grid_sz <= MAX_BLOCK_SZ) {
            clearBuffer(commandList, m_dummy_blocks_sums->resource(), 1);
            gpuPrescan(commandList, d_block_sums->resource(), d_block_sums->resource(), m_dummy_blocks_sums->resource(), 1, grid_sz);
        } else {
            auto d_in_block_sums = m_in_block_sums.at(grid_sz);
            commandList->CopyResource(d_in_block_sums->resource().Get(), d_block_sums->resource().Get());
            run_(commandList, d_block_sums->resource(), d_in_block_sums->resource(), grid_sz);
        }

        gpuAddBlockSums(commandList, d_out, d_block_sums->resource(), grid_sz, numElems);
    }

    void clearBuffer(ComPtr<ID3D12GraphicsCommandList> commandList, ComPtr<ID3D12Resource> buffer, unsigned int numElems) {
        const int block_size = 256; // matches shader!
        const int grid_size = (numElems + block_size - 1) / block_size;
        scan_clear_buffer_hlsl::constants c{ static_cast<int>(numElems) };
        m_shader_clear_buffer->run(commandList, { grid_size }, { buffer }, &c);
    }

    void gpuPrescan(ComPtr<ID3D12GraphicsCommandList> commandList, ComPtr<ID3D12Resource> d_out, ComPtr<ID3D12Resource> d_in, ComPtr<ID3D12Resource> d_block_sums, unsigned int grid_sz, unsigned int numElems) {
        scan_prescan_hlsl::constants c{ static_cast<int>(numElems) };
        m_shader_prescan->run(commandList, { grid_sz }, { d_out, d_in, d_block_sums }, &c);
    }

    void gpuAddBlockSums(ComPtr<ID3D12GraphicsCommandList> commandList, ComPtr<ID3D12Resource> d_out, ComPtr<ID3D12Resource> d_block_sums, unsigned int grid_sz, unsigned int numElems) {
        scan_add_block_sums_hlsl::constants c{ static_cast<int>(numElems) };
        m_shader_add_block_sums->run(commandList, { grid_sz }, { d_out, d_block_sums }, &c);
    }

private:
    const unsigned int m_n_els;
};
