using StaticArrays
using MOTCore: get_pos

const S2V = SVector{2, Float64}

function target_distractor_iou(state)
    positions = map(get_pos, state.objects)
    polygon_bbox_iou(positions[1:4], positions[5:8])
end

function polygon_bbox_iou(a::Vector{S2V}, b::Vector{S2V})
    bbox1 = compute_bounding_box(a)
    bbox2 = compute_bounding_box(b)
    bbox_iou(bbox1, bbox2)
end

function polygon_bbox_iou(a::Vector{S2V}, b::Vector{S2V})
    bbox1 = compute_bounding_box(a)
    bbox2 = compute_bounding_box(b)
    bbox_iou(bbox1, bbox2)
end

"""
Compute Intersection over Union (IoU) for two bounding boxes.
IoU = Area(intersection) / Area(union)
"""
function bbox_iou(bbox1, bbox2)
    intersection = bbox_intersection(bbox1, bbox2)

    if intersection === nothing
        return 0.0  # No intersection
    end

    intersection_area = bbox_area(intersection)
    union_area = bbox_union_area(bbox1, bbox2)

    iszero(union_area) ? 0.0 : intersection_area / union_area
end

"""
Compute the intersection of two axis-aligned bounding boxes.
Returns the intersection bounding box or nothing if no intersection.
"""
function bbox_intersection(bbox1, bbox2)
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2

    # Find intersection bounds
    inter_min_x = max(min_x1, min_x2)
    inter_min_y = max(min_y1, min_y2)
    inter_max_x = min(max_x1, max_x2)
    inter_max_y = min(max_y1, max_y2)

    # Check if intersection exists
    if inter_min_x >= inter_max_x || inter_min_y >= inter_max_y
        return nothing  # No intersection
    end

    return (inter_min_x, inter_min_y, inter_max_x, inter_max_y)
end


"""
Compute the union area of two bounding boxes.
Union area = Area(bbox1) + Area(bbox2) - Area(intersection)
"""
function bbox_union_area(bbox1, bbox2)
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)

    intersection = bbox_intersection(bbox1, bbox2)
    if intersection === nothing
        return area1 + area2
    end

    intersection_area = bbox_area(intersection)
    return area1 + area2 - intersection_area
end

"""
Compute the area of a bounding box.
bbox format: (min_x, min_y, max_x, max_y)
"""
function bbox_area(bbox)
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    return max(0.0, width * height)
end

"""
Compute the axis-aligned bounding box for a polygon given its vertices.
Returns (min_x, min_y, max_x, max_y).
"""
function compute_bounding_box(vertices::Vector{S2V})
    n = length(vertices)
    min_x = Inf
    min_y = Inf
    max_x = -Inf
    max_y = -Inf
    @inbounds for i = 1:n
        x, y = vertices[i]
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    end
    return (min_x, min_y, max_x, max_y)
end
