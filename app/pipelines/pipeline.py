import copy
import os
import itertools
from typing import List
import numpy as np
NaN = np.nan
from scipy.spatial import distance
import numpy as np
import pandas as pd

from .dataset import Dataset
from .predicateitem import JoinParameters, PredicateItem, PredicateItemGroup
from .tools.operator_logging import loggable_operator


class Pipeline:
    SLASH_ESCAPER = "--escaped_slash--"

    def __init__(self, database_name: str, initial_collection_names: List[str], data_folder="./data", discrete_categories_count=20, exploration_columns=[]):
        self.initial_collection_names = initial_collection_names
        self.database_name = database_name
        self.discrete_categories_count = discrete_categories_count
        self.data_folder = data_folder
        self.database_folder = f"{self.data_folder}/{database_name}"
        self.column_forced_types = pd.read_csv(
            f"{self.database_folder}/column_forced_types.csv")
        self.initial_collection = self.load_table(
            self.initial_collection_names[0])
        if os.path.isfile(f"{self.database_folder}/foreign_keys.csv"):
            self.foreign_keys = pd.read_csv(
                f"{self.database_folder}/foreign_keys.csv")

        all_columns_to_discretize = pd.read_csv(
            f"{self.database_folder}/columns_to_discretize.csv")
        self.columns_to_discretize = all_columns_to_discretize[all_columns_to_discretize.table.isin(
            initial_collection_names)]
        if len(self.columns_to_discretize) > 0:
            self.columns_to_discretize = self.columns_to_discretize.apply(lambda column: f"{column.table}.{column.column}", axis=1).to_list()


        joined_tables = self.initial_collection_names[:1]
        self.initial_joins: List[JoinParameters] = []
        tables_to_join = self.initial_collection_names[1:]
        for target_collection_name in tables_to_join:
            relations = self.foreign_keys[(
                (self.foreign_keys["table1"].isin(joined_tables)) & (self.foreign_keys["table2"] == target_collection_name)) | ((
                    self.foreign_keys["table2"].isin(joined_tables)) & (self.foreign_keys["table1"] == target_collection_name))]
            if len(relations) == 1:
                relation = relations.iloc[0]
                joined_tables.append(target_collection_name)
                target_collection = self.load_table(target_collection_name)
                if target_collection_name == relation.table1:
                    self.initial_joins.append(JoinParameters(target_collection_name=relation.table1,
                                                             left_attribute=f"{relation.table2}.{relation.attribute2}", right_attribute=f"{relation.table1}.{relation.attribute1}", other_collection=relation.table2))
                    self.initial_collection = pd.merge(self.initial_collection, target_collection, how="inner", left_on=f"{relation.table2}.{relation.attribute2}",
                                                       right_on=f"{relation.table1}.{relation.attribute1}", sort=False, suffixes=('', f"_{target_collection_name}"))
                else:
                    self.initial_joins.append(JoinParameters(target_collection_name=relation.table2,
                                                             left_attribute=f"{relation.table1}.{relation.attribute1}", right_attribute=f"{relation.table2}.{relation.attribute2}", other_collection=relation.table1))
                    self.initial_collection = pd.merge(self.initial_collection, target_collection, how="inner", left_on=f"{relation.table1}.{relation.attribute1}",
                                                       right_on=f"{relation.table2}.{relation.attribute2}", sort=False, suffixes=('', f"_{target_collection_name}"))
            elif len(relations) == 0:
                if tables_to_join.index(target_collection_name) == len(tables_to_join) - 1:
                    raise Exception(
                        "Unable to join " + target_collection_name)
                tables_to_join.append(target_collection_name)
            else:
                raise Exception(
                    "Multiple relations possible to join " + target_collection_name)


        self.exploration_columns = exploration_columns
        if len(self.exploration_columns) == 0:


            self.exploration_columns = list(self.initial_collection.columns)

        self.ordered_dimensions = {}
        for column_name in self.columns_to_discretize:
            if column_name in self.initial_collection.columns.values:
                self.initial_collection[column_name] = pd.qcut(
                    self.initial_collection[column_name], self.discrete_categories_count, duplicates="drop")
                self.ordered_dimensions[column_name] = list(map(lambda cat: str(
                    cat), self.initial_collection[column_name].dtype.categories))

        self.interesting_attributes = Pipeline.find_interesting_attributes(
            self.initial_collection)

    @classmethod
    def find_interesting_attributes(cls, dataframe, count=None):
        interesting_attributes = []
        na_ratio = dataframe.isna().sum()/len(dataframe)
        density_ratio = dataframe.nunique() / len(dataframe)
        attributes = pd.DataFrame(
            data={"attribute": dataframe.columns, "na_ratio": na_ratio, "density_ratio": density_ratio})
        attributes = attributes.sort_values(by=["density_ratio", "na_ratio"])
        if count == None:
            interesting_attributes = attributes[(attributes.na_ratio < 0.3) & (
                attributes.density_ratio < 0.3)].attribute.tolist()
            if len(interesting_attributes) == 0:
                interesting_attributes = dataframe.columns
            return interesting_attributes
        else:
            return attributes.iloc[0:count].attribute.tolist()


    def load_table(self, table_name):
        csv_path = f"{self.database_folder}/{table_name}.csv"


        forced_type_columns = self.column_forced_types[self.column_forced_types.table == table_name]
        dtype_map = {}
        for _, column in forced_type_columns.iterrows():
            dtype_map[column.column] = column.type


        if table_name == "galaxies":
            dtype_map["objID"] = "string"


        df = pd.read_csv(csv_path, dtype=dtype_map) if dtype_map else pd.read_csv(csv_path)

        return df.add_prefix(table_name + ".")


    @loggable_operator()
    def by_filter(self, dataset: Dataset, predicate_item: PredicateItem, logger=None):
        dataset.predicate.append(predicate_item)
        dataset.data = self.query_predicate_item(dataset, predicate_item)
        return dataset

    @loggable_operator()
    def by_facet(self, dataset: Dataset, attributes: List[str], number_of_groups: int, return_datasets=True, logger=None):
        facets = dataset.data.groupby(attributes, observed=True)
        grouped_sets = facets.size().to_frame('counts')
        top_groups = grouped_sets.nlargest(number_of_groups, "counts")
        if return_datasets == False:
            return top_groups
        else:
            datasets = []
            for index, group in top_groups.iterrows():
                new_dataset = copy.deepcopy(dataset)
                for i in range(len(attributes)):
                    if len(attributes) == 1:
                        new_dataset.predicate.append(PredicateItem(attributes[0], "==", index, is_category=str(
                            dataset.data[attributes[0]].dtype) == "category"))
                    else:
                        new_dataset.predicate.append(PredicateItem(attributes[i], "==", index[i], is_category=str(
                            dataset.data[attributes[i]].dtype) == "category"))
                new_dataset.data = self.query_predicate(
                    new_dataset, new_dataset.predicate)
                datasets.append(new_dataset)
            return datasets


    @loggable_operator()
    def by_join(self, dataset: Dataset, target_collection_name: str, left_attribute: str, right_attribute: str, other_collection: str = None, drop_column="", logger=None):

        target_collection = self.load_table(target_collection_name)
        join_parameters = JoinParameters(
            target_collection_name, left_attribute, right_attribute, other_collection=other_collection)
        dataset.joins.append(join_parameters)
        dataset.data = pd.merge(dataset.data, target_collection, how="inner", left_on=join_parameters.left_attribute,
                                right_on=join_parameters.right_attribute, sort=False, suffixes=('', f"_{target_collection_name}"))
        if drop_column == "both":
            dataset.data = dataset.data.drop(
                columns=[join_parameters.left_attribute, join_parameters.right_attribute])
        if drop_column == "left":
            dataset.data = dataset.data.drop(
                columns=[join_parameters.left_attribute])
        if drop_column == "right":
            dataset.data = dataset.data.drop(
                columns=[join_parameters.right_attribute])
        return dataset

    @loggable_operator()
    def by_neighbors(self, dataset: Dataset, attributes: List[str], logger=None):

        if len(attributes) == 0:
            attributes = map(
                lambda item: item.attribute in self.ordered_dimensions, dataset.predicate.components)

        results = [dataset]
        for attribute in attributes:
            if attribute in self.ordered_dimensions:
                predicate_item = next(
                    (item for item in dataset.predicate.components if item.attribute == attribute), None)
                if predicate_item != None:
                    index = self.ordered_dimensions[attribute].index(
                        str(predicate_item.value))
                    if index != 0:
                        new_set = self.get_neighboring_set(
                            dataset, attribute, index - 1)
                        if len(new_set.data)>0:
                            results.append(new_set)
                    if index + 1 != len(self.ordered_dimensions[attribute]):
                        new_set = self.get_neighboring_set(
                            dataset, attribute, index + 1)
                        if len(new_set.data)>0:
                            results.append(new_set)
        return results

    @loggable_operator()
    def by_distribution(self, dataset: Dataset, logger=None):
        ordered_attributes_in_description=[ x for x in dataset.predicate.get_attributes() if x in self.ordered_dimensions]
        if len(ordered_attributes_in_description) > 1:
            dataset_vector = []
            for attribute in ordered_attributes_in_description:
                value = dataset.predicate.get_filter_values(attribute)[0]
                dataset_vector.append(self.ordered_dimensions[attribute].index(str(value)))

            result_vectors = []
            new_vector = [x+1 for x in dataset_vector]
            while all([x < self.discrete_categories_count for x in new_vector]):
                result_vectors.append(new_vector)
                new_vector = [x+1 for x in new_vector]

            new_vector = [x-1 for x in dataset_vector]
            while all([x >= 0 for x in new_vector]):
                result_vectors.append(new_vector)
                new_vector = [x-1 for x in new_vector]

            if len(result_vectors) == 0:
                return [dataset]

            result_sets = []
            for vector in result_vectors:
                new_dataset = dataset.copy()
                for i in range(len(ordered_attributes_in_description)):
                    attribute = ordered_attributes_in_description[i]
                    new_dataset.predicate.remove_attribute(attribute)
                    new_dataset.predicate.append(PredicateItem(attribute, "==", self.ordered_dimensions[attribute][vector[i]], is_category=True))
                self.reload_set_data(dataset=new_dataset, apply_joins=True, apply_predicate=True)
                result_sets.append(new_dataset)

            return result_sets


        else:
            return [dataset]


    def get_neighboring_set(self, dataset: Dataset, attribute: str, value_index: int):
        new_dataset = Dataset(predicate=copy.deepcopy(
            dataset.predicate), joins=copy.deepcopy(dataset.joins))
        predicate_item = next(
            (item for item in new_dataset.predicate.components if item.attribute == attribute), None)
        predicate_item.value = self.ordered_dimensions[attribute][value_index]
        self.reload_set_data(new_dataset, apply_joins=True,
                             apply_predicate=True)
        return new_dataset

    def calculate_overlap(self, set1: set, set2: set):
        return len(set1 & set2) / len(set1 | set2)

    def reload_set_data(self, dataset: Dataset, apply_joins=False, apply_predicate=False, initialize_data=True):
        if initialize_data:
            dataset.data = self.initial_collection[:]
        if apply_joins:
            for join_parameters in dataset.joins:
                target_collection = self.load_table(
                    join_parameters.target_collection_name)
                dataset.data = pd.merge(
                    dataset.data,
                    target_collection,
                    how="inner",
                    left_on=join_parameters.left_attribute,
                    right_on=join_parameters.right_attribute,
                    sort=False,
                    suffixes=(
                        "", f"_{join_parameters.target_collection_name}"),
                )
        else:
            dataset.joins = []
        if apply_predicate and not dataset.predicate.is_empty():
            dataset.data = self.query_predicate(dataset, dataset.predicate)
        else:
            dataset.predicate = PredicateItemGroup("&")
        return dataset

    def by_superset(self, dataset: Dataset, set_ids_to_ignore=[], number_of_sets_to_return=3):
        if len(dataset.predicate.components) == 0:
            return dataset
        else:
            original_count = len(dataset.data)
            print(f"Original set item count: {original_count}")
            smallest_item_count = len(self.initial_collection)
            new_sets = []
            predicate_attributes = dataset.predicate.get_attributes()
            if len(predicate_attributes) == 1:
                raise Exception(
                    "Superset impossible, only one attribute is filtered")
            else:
                candidates = []
                for attribute in predicate_attributes:
                    new_dataset = Dataset(predicate=copy.deepcopy(
                        dataset.predicate), joins=copy.deepcopy(dataset.joins))
                    new_dataset.predicate.remove_attribute(attribute)
                    self.reload_set_data(
                        new_dataset, apply_joins=True, apply_predicate=True)
                    candidates.append(new_dataset)

                candidates = sorted(candidates, key=lambda candidate: len(candidate.data), reverse=True)
                return candidates[0:number_of_sets_to_return]


    def by_overlap(self, dataset: Dataset, number_of_groups=3, max_seconds=1, return_datasets=True, logger=None):
        raise NotImplementedError()

    def get_dataset(self):
        return Dataset(data=self.initial_collection[:])

    def query_predicate_item(self, dataset: Dataset, predicate_item: PredicateItem):
        value = predicate_item.value
        if type(value) == str:
            value = value.replace('\n', '\\n')
        return dataset.data[dataset.data[predicate_item.attribute] == value]

    def query_predicate(self, dataset: Dataset, predicate: PredicateItemGroup):
        from revision_support.query_backends import GALAXY_BATCH
        batch = GALAXY_BATCH.get()
        if batch is not None:
            dataset.data = batch.filter(self, dataset.data, predicate)
            return dataset.data
        for item in predicate.components:
            if item.is_multiple_line_string:
                dataset.data = self.query_predicate_item(dataset, item)
            elif item.is_category:
                index = self.ordered_dimensions[item.attribute].index(
                    str(item.value))
                value = self.initial_collection[item.attribute].dtype.categories[index]
                dataset.data = dataset.data[dataset.data[item.attribute] == value]
            else:
                dataset.data = dataset.data[dataset.data[item.attribute]
                                            == item.value]
        return dataset.data
