#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    //
    // Backend buffer
    //

    // buffer type
    typedef void * ggml_backend_buffer_type_context_t;

    struct ggml_backend_buffer_type_i {
        const char *          (*GGML_CALL get_name)        (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*GGML_CALL alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*GGML_CALL get_alignment)   (ggml_backend_buffer_type_t buft);
        // max buffer size that can be allocated
        size_t                (*GGML_CALL get_max_size)    (ggml_backend_buffer_type_t buft);
        // data size needed to allocate the tensor, including padding
        size_t                (*GGML_CALL get_alloc_size)  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
        // check if tensor data is in host memory
        bool                  (*GGML_CALL is_host)         (ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_buffer_type_context_t context;
    };

    // buffer
    typedef void * ggml_backend_buffer_context_t;

    struct ggml_backend_buffer_i {
        const char * (*GGML_CALL get_name)   (ggml_backend_buffer_t buffer);
        void         (*GGML_CALL free_buffer)(ggml_backend_buffer_t buffer);
        void *       (*GGML_CALL get_base)   (ggml_backend_buffer_t buffer);
        void         (*GGML_CALL init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        void         (*GGML_CALL set_tensor) (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*GGML_CALL get_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool         (*GGML_CALL cpy_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
        void         (*GGML_CALL clear)      (ggml_backend_buffer_t buffer, uint8_t value);
        void         (*GGML_CALL reset)      (ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
    };

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        ggml_backend_buffer_context_t context;
        size_t size;
        enum ggml_backend_buffer_usage usage;
    };

    GGML_CALL ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t      buft,
            struct ggml_backend_buffer_i           iface,
                   ggml_backend_buffer_context_t   context,
                   size_t                          size);

    // do not use directly, use ggml_backend_tensor_copy instead
    bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);

    // buffer that contains a collection of buffers
    GGML_CALL ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
    GGML_CALL bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    GGML_CALL void                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);

    //
    // Backend
    //

    typedef void * ggml_backend_context_t;

    // ggml_backend_i 结构体定义了一组函数指针，这些函数指针用于实现不同后端的计算与内存管理功能
    struct ggml_backend_i {
        const char * (*GGML_CALL get_name)(ggml_backend_t backend);  // 获取后端的名称

        void (*GGML_CALL free)(ggml_backend_t backend);     // 释放后端资源

        // buffer allocation 获取后端的默认缓冲区类型，帮助在不同后端之间进行内存管理
        ggml_backend_buffer_type_t (*GGML_CALL get_default_buffer_type)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access 异步设置张量的数据，可以在不阻塞主线程的情况下更新张量的数据
        void (*GGML_CALL set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size); 
        // 异步获取张量的数据，允许在后台获取数据而不影响主线程的运行
        void (*GGML_CALL get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);  
        // 异步复制张量的数据，支持源后端与目标后端之间的数据传输
        bool (*GGML_CALL cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst); 

        // (optional) complete all pending operations
        void (*GGML_CALL synchronize)(ggml_backend_t backend); // 完成所有挂起的操作，确保所有异步操作都已完成

        // compute graph with a plan (not used currently)
        // create a new plan for a graph
        ggml_backend_graph_plan_t (*GGML_CALL graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph); 
        // 释放计算计划的资源
        void                      (*GGML_CALL graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan); 
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        // 更新已有的计算计划，以便在保持拓扑结构不变的情况下提高效率
        void                      (*GGML_CALL graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);   
        // compute the graph with the plan  使用计划计算图，执行计算
        enum ggml_status          (*GGML_CALL graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);      

        // compute graph without a plan (async)
        enum ggml_status (*GGML_CALL graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph); //执行计算图

        // check if the backend can compute an operation
        bool (*GGML_CALL supports_op)(ggml_backend_t backend, const struct ggml_tensor * op);  

        // check if the backend can use tensors allocated in a buffer type
        bool (*GGML_CALL supports_buft)(ggml_backend_t backend, ggml_backend_buffer_type_t buft);

        // check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
        // these should be expensive operations with large batch sizes that may benefit from running on this backend
        // even if the weight has to be copied from the CPU temporarily
        bool (*GGML_CALL offload_op)(ggml_backend_t backend, const struct ggml_tensor * op);

        // (optional) event synchronization
        // create a new event that can record events on this backend instance
        ggml_backend_event_t (*GGML_CALL event_new)         (ggml_backend_t backend);  // 创建一个新的事件，可以用于记录后端实例中的事件
        void                 (*GGML_CALL event_free)        (ggml_backend_event_t event);   // 释放事件资源
        // record an event on the backend instance that created it
        void                 (*GGML_CALL event_record)      (ggml_backend_event_t event);   // 在创建该事件的后端实例中记录事件
        // wait for an event on on a different backend instance
        void                 (*GGML_CALL event_wait)        (ggml_backend_t backend, ggml_backend_event_t event); // 在不同后端实例上等待事件
        // block until an event is recorded
        void                 (*GGML_CALL event_synchronize) (ggml_backend_event_t event); // 阻止直到事件被记录，确保操作的同步性
    };

    struct ggml_backend {
        ggml_guid_t guid;

        struct ggml_backend_i iface;
        ggml_backend_context_t context;
    };

    struct ggml_backend_event {
        ggml_backend_t backend;
        void * context;
    };

    //
    // Backend registry
    //

    typedef ggml_backend_t (*GGML_CALL ggml_backend_init_fn)(const char * params, void * user_data);

    GGML_CALL void ggml_backend_register(const char * name, ggml_backend_init_fn init_fn, ggml_backend_buffer_type_t default_buffer_type, void * user_data);

#ifdef  __cplusplus
}
#endif
