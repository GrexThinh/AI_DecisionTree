import(Pkg)
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("StatsBase")

using StatsBase
using CSV
using Pkg
using DataFrames

# Điều kiện dừng
MINIMUM_SAMPLE_SIZE = 4
MAX_TREE_DEPTH = 4

# Lấy tập dữ liệu từ file .csv
df=DataFrame(CSV.File("iris.csv"))

# Đọc dữ liệu tập hoa Iris lưu vào 1 dạng danh sách gồm các phần tử kiểu dict
function readIrisData(df)
    dataset = []
    for row in eachrow(df)
        instance = Dict()
        instance["sepal_length"] = float(row[1])
        instance["sepal_width"] = float(row[2])
        instance["petal_length"] = float(row[3])
        instance["petal_width"] = float(row[4])
        instance["species"] = row[5]
        push!(dataset, instance)
    end
    return dataset
end

# Tính số entropy của tập thuộc tính đích các loài hoa 
function calEntropy(dataset)
    if size(dataset) == 0
        return 0
    end
    target_attribute_name = "species"
    target_attribute_values = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    data_entropy = 0
    for val in target_attribute_values
        p = length([elem for elem in dataset if elem[target_attribute_name] == val]) / length(dataset)
        if p > 0
            data_entropy += -p * log(2, p)
        end
    end
    return data_entropy
end

# Tính giá trị entropy trung bình trên từng thuộc tính 
# Chia giá trị của thuộc tính thành 2 phần
function information_gain(attribute_name, split, dataset)

    # Chia tập dữ liệu và tính xác suất từng thành phần trong từng tập
    set_smaller = [elem for elem in dataset if elem[attribute_name] < split]
    p_smaller = length(set_smaller) / length(dataset)
    set_greater_equals = [elem for elem in dataset if elem[attribute_name] >= split]
    p_greater_equals = length(set_greater_equals) / length(dataset)

    # Tính giá trị information gain
    info_gain = calEntropy(dataset)
    info_gain -= p_smaller * calEntropy(set_smaller)
    info_gain -= p_greater_equals * calEntropy(set_greater_equals)

    return info_gain
end

# get criterion and optimal split to maximize information gain
# Chọn thuộc tính và lấy giá trị phân chia
function max_information_gain(attribute_list, attribute_values, dataset)
    max_info_gain = 0
    max_info_gain_attribute=0
    max_info_gain_split=0
    for attribute in attribute_list # Kiểm tra tất cả các thuộc tính đầu vào
        for split in attribute_values[attribute] # Kiểm tra tất cả các giá trị có thể để phân chia giới hạn
            split_info_gain = information_gain(attribute, split, dataset) # Tính giá trị information gain
            if split_info_gain >= max_info_gain
                max_info_gain = split_info_gain
                max_info_gain_attribute = attribute
                max_info_gain_split = split
            end
        end
    end
    return max_info_gain, max_info_gain_attribute, max_info_gain_split
end

# Tạo cấu trúc cây có các thuộc tính trên từng node
mutable struct TreeNode
    is_leaf
    dataset
    split_attribute
    split
    attribute_list
    attribute_values
    left_child 
    right_child
    prediction
    depth
end

function buildTree(self::TreeNode)
    training_set=self.dataset

    # Xây dựng cây nếu điều kiện dừng không thỏa
    if self.depth < MAX_TREE_DEPTH && length(training_set) >= MINIMUM_SAMPLE_SIZE && length(Set([elem["species"] for elem in training_set])) > 1
        # Lấy thuộc tính và chia với giá trị information gain cao nhất
        max_gain, attribute, split = max_information_gain(self.attribute_list, self.attribute_values, training_set)
        # Kiểm tra nếu information gain lớn hơn 0 (một điều kiện dừng khác)
        if max_gain > 0
            # Phân chia cây
            self.split = split
            self.split_attribute = attribute
            
            # Tạo node con
            training_set_l = [elem for elem in training_set if elem[attribute] < split]
            training_set_r = [elem for elem in training_set if elem[attribute] >= split]
            self.left_child = TreeNode(false, training_set_l, nothing, nothing, self.attribute_list, self.attribute_values, nothing, nothing, nothing, self.depth + 1)
            self.right_child = TreeNode(false, training_set_r, nothing, nothing, self.attribute_list, self.attribute_values, nothing, nothing, nothing, self.depth + 1)
            buildTree(self.left_child)
            buildTree(self.right_child)
        else
            self.is_leaf=true
        end
    else
        self.is_leaf=true
    end

    if self.is_leaf
        # Dự đoán node lá là phổ biến nhất trong tập training
        setosa_count = versicolor_count = virginica_count = 0
        for elem in training_set
            if elem["species"] == "Iris-setosa"
                setosa_count += 1
            elseif elem["species"] == "Iris-versicolor"
                versicolor_count += 1
            else
                virginica_count += 1
            end
        end
        dominant_class = "Iris-setosa"
        dom_class_count = setosa_count
        if versicolor_count >= dom_class_count
            dom_class_count = versicolor_count
            dominant_class = "Iris-versicolor"
        end
        if virginica_count >= dom_class_count
            dom_class_count = virginica_count
            dominant_class = "Iris-virginica"
        end
        self.prediction = dominant_class
    end
end

# Kiểm tra độ đo accuracy trên cây quyết định
function predictValue(self::TreeNode, sample)
    if self.is_leaf==true
        return self.prediction
    else
        if sample[self.split_attribute] < self.split
            return predictValue(self.left_child,sample)
        else
            return predictValue(self.right_child,sample)
        end
    end
end

# Hợp nhất 2 node lá nếu chúng có cùng giá trị dự đoán
function mergeLeaves(self::TreeNode)
    if self.is_leaf==false
        mergeLeaves(self.left_child)
        mergeLeaves(self.right_child)
        if self.left_child.is_leaf==true && self.right_child.is_leaf==true && self.left_child.prediction == self.right_child.prediction
            self.is_leaf = true
            self.prediction = self.left_child.prediction
        end
    end
end

# In cây quyết định
function printTree(self::TreeNode, prefix)
    if self.is_leaf
        if self.depth==1
            Base.print("\t" * "[" * string(self.depth) * "] " * prefix * self.prediction * "\n")
        elseif self.depth==2
            Base.print("\t\t" * "[" * string(self.depth) * "] " * prefix * self.prediction * "\n")
        elseif self.depth==3
            Base.print("\t\t\t" * "[" * string(self.depth) * "] " * prefix * self.prediction * "\n")
        else
            Base.print("\t\t\t\t" * "[" * string(self.depth) * "] " * prefix * self.prediction * "\n")
        end
    else
        if self.depth==0
            Base.print("[" * string(self.depth) *  "]"  * prefix * " " * self.split_attribute * " < " * string(self.split) * "?" * "\n")
        elseif self.depth==1
            Base.print("\t" * "[" * string(self.depth) * "] " * prefix * self.split_attribute * " < " * string(self.split) * "?" * "\n")
        elseif self.depth==2
            Base.print("\t\t" * "[" * string(self.depth) * "] " * prefix * self.split_attribute * " < " * string(self.split) * "?" * "\n")
        elseif self.depth==3
            Base.print("\t\t\t" * "[" * string(self.depth) * "] " * prefix * self.split_attribute * " < " * string(self.split) * "?" * "\n")
        else
            Base.print("\t\t\t\t" * "[" * string(self.depth) * "] " * prefix * self.split_attribute * " < " * string(self.split) * "?" * "\n")
        end
        printTree(self.left_child,"if True: ")
        printTree(self.right_child,"if False: ")
    end
end

# Tạo cấu trúc cây quyết định ID3
mutable struct DecisionTreeID3
    root
end

# Xây dựng cây quyết định ID3
function buildID3(self::DecisionTreeID3, training_set, attribute_list, attribute_values)
    self.root=TreeNode(false,training_set,nothing,nothing,attribute_list,attribute_values,nothing,nothing,nothing,0)
    buildTree(self.root)
end

# Hợp nhất các lá có cùng giá trị dự đoán trên cây quyết định ID3
function mergeLeavesID3(self::DecisionTreeID3)
    mergeLeaves(self.root)
end

# Tính độ đo accuracy trên cây quyết định ID3
function predictValueID3(self::DecisionTreeID3, sample)
    return predictValue(self.root, sample)
end

# In cây quyết định ID3
function printID3(self::DecisionTreeID3)
    printTree(self.root,"")
end

# Hàm main
function main()
    dataset=readIrisData(df)

    test_set=[]
    training_set=[]
    # Lấy ngẫu nhiên 1/3 tập dữ liệu để test trong tập dữ liệu ban đầu
    test_set = sample(dataset, trunc(Int,(1/3 * length(dataset))))
    # Lấy 2/3 tập dữ liệu còn lại để training
    for i in dataset
        check=true
        if i in test_set
            check=false
        end
        if check==true
            push!(training_set,i)
            if length(training_set)==trunc(Int,(2/3 * length(dataset)))
                break
            end
        end
    end
    
    println("Iris data set size: ", length(dataset))
    println("Training set size: ", length(training_set))
    println("Test set size: ", length(test_set))

    # danh sách tất cả các thuộc tính đầu vào của hoa
    attr_list = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    # Lấy danh sách tất cả các giá trị thuộc tính hợp lệ
    attr_domains = Dict()
    list_attr=[]
    list_attr=keys(dataset[1])
    for attr in list_attr
        attr_domain = Set()
        for data in dataset
            push!(attr_domain,data[attr])
        end
        list_attr_domain=[]
        list_attr_domain=attr_domain
        attr_domains[attr]=list_attr_domain
    end

    # Xây dựng cây quyết định ID3
    dt=DecisionTreeID3(Nothing)
    buildID3(dt, dataset, attr_list, attr_domains)
    mergeLeavesID3(dt)

    Base.print("\n\t\t\t\t\t\t\t\t ID3 DECISION TREE \n")
    printID3(dt)
    Base.print("\n")

    # Tính giá trị độ đo accuracy trên tập test
    accuracy = 0
    for sample in test_set
        if sample["species"] == predictValueID3(dt, sample)
            accuracy += (1/length(test_set))
        end
    end
    result=round(accuracy*100, digits=2)
    Base.print("==> Accuracy result of test set: " * string(result) * "%")
end

# Chạy chương trình
main()